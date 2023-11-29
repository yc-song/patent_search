# Create 'keys.json' file in ML folder consists of following three lines:
# {
#     "OpenAI_API_KEY":"your openai api key"
# }

import os, json, re
import pandas as pd
from langchain_experimental.agents.agent_toolkits.csv.base import create_pandas_dataframe_agent
# from langchain.llms import OpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, AgentExecutor, load_tools, initialize_agent
from langchain.agents.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

with open("backend/ML/keys.json", 'r') as f:
    keys = json.load(f)
OPENAI_API_KEY = keys["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def textRAG_from_ids(df, ids, user_query, max_rows = 3):
    ids = list(set(ids))
    minimized_df = df[['출원번호','요약', '청구항']].iloc[ids].reset_index()
    minimized_df = preprocess_df(minimized_df)
    # print("minimized_df:")
    # print(minimized_df)
    max_rows = max_rows
    chunks = list(chunk_data_frame(minimized_df, max_rows))
    results = []

    prompt_developers_template = """
    너의 데이터베이스는 특허 파일이야.
    아래의 내용을 참고하여 질문에 답해줘
    {user_query}
    """
    prompt_template = PromptTemplate.from_template(prompt_developers_template)

    for i, chunk in enumerate(chunks):

        agent = create_pandas_dataframe_agent(
                                ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"),
                                # model,
                                chunk, 
                                verbose=True,
                                reduce_k_below_max_tokens=True,
                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                handle_parsing_errors=True
                                )
    # Run the CSV Agent on the chunk

        result = agent.run(prompt_template.format(user_query = user_query))

        results.append(result)
        # print(f"chunk {i+1} of {len(chunks)} ended.")

    # revised_result = revise_result(results)
    return results

def textRAG_from_one_id_detail(df, id, user_query):
    minimized_df = df.iloc[id:1+id].reset_index()

    chunks = chunk_oneRow_data_frame(minimized_df)
    results = []

    prompt_developers_template = """
    너의 데이터베이스는 특허 데이터야.
    아래의 내용을 참고하여 질문에 답해줘
    {user_query}
    원하는 내용이 없다면 대답하지 않아도 돼.
    """
    prompt_template = PromptTemplate.from_template(prompt_developers_template)

    for i, chunk in enumerate(chunks):

        agent = create_pandas_dataframe_agent(
                                ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"),
                                # model,
                                chunk, 
                                verbose=True,
                                reduce_k_below_max_tokens=True,
                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                handle_parsing_errors=True
                                )
    # Run the CSV Agent on the chunk

        result = agent.run(prompt_template.format(user_query = user_query))

        results.append(result)
        # print(f"chunk {i+1} of {len(chunks)} ended.")

    # revised_result = revise_result(results)
    return results


def chunk_data_frame(df, chunk_size):

    num_chunks = len(df) // chunk_size + (len(df) % chunk_size > 0)

    for i in range(num_chunks):

        yield df[i * chunk_size:(i + 1) * chunk_size]

def chunk_oneRow_data_frame(df):
    chunks = []
    for i in range(len(df.columns)):
        if len(str(df.iloc[0,i])) < 10:
            chunks.append(pd.DataFrame(df.iloc[0:1, i]))
    return chunks


def revise_result(results):
    revised = {}
    for r in results:
        if '출원번호' in r:
            id = extract_id(r)
            revised['id'] = id
    return revised

def extract_id(text):
    start = text.find('출원번호') + 3
    num_start = start

    # extract the starting point of id
    while True:
        try:
            is_num = int(text[num_start])
            break            
        except:
            num_start +=1

    num_end = num_start
    # extract the end point of id
    while True:
        try:
            is_num = int(text[num_end])
            num_end +=1
        except:
            break

    return int(text[num_start:num_end])

def individual_chatRAG2(df, id, user_query, agent = None, history = []):
    if agent == None:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
        special_df = df.loc[id:id, ['출원번호','요약']]
        tools = load_tools(["requests_all"], llm=llm)
        # print("special_df:")
        # print(special_df)
        DataFrame_agent = create_pandas_dataframe_agent(
            llm=llm, 
            df=special_df,
            # verbose=True,
            # agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            reduce_k_below_max_tokens=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            memory=ConversationBufferMemory()
            )
        DataFrame_tool = Tool(
            name="DataFrame Agent",
            func=DataFrame_agent.run,
            description="너가 보고 있는 데이터는 특허 데이터야. 특허 관련 질문은 이 데이터를 참고해서 대답해줘."
        )
        tools.extend([DataFrame_tool])
        agent = initialize_agent(
            llm=llm,
            tools=tools,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
            memory=ConversationBufferMemory(),
        )


    response = agent.run(user_query)
    print("response:", response)

    return response, agent

def preprocess_df(df):
    if '청구항' not in df.columns:
        return df

    nrow, ncol = df.shape
    for r in range(nrow):
        claim_text = df.loc[r, '청구항']
        df.loc[r, '청구항'] = claim_text_process(claim_text)

    return df

def claim_text_process(text):
    # 1. replace parenthesis part into empty string.
    text = replace_enclosed_integers(text)
    text = truncate_sentence(text)
    return text


def replace_enclosed_integers(text):
    # Define a regular expression pattern to match integers enclosed in small parenthesis
    pattern = r'\((\d+)\)'

    # Use re.sub to find all matches and replace them with a specific string (e.g., 'REPLACEMENT_STRING')
    replaced_text = re.sub(pattern, '', text)

    return replaced_text

def truncate_sentence(sentence):
    # Split the sentence into words
    words = sentence.split()

    # Consider up to the first 1000 words
    truncated_sentence = ' '.join(words[:700])

    return truncated_sentence

def individual_chatRAG(df, id, user_query, agent = None, history = []):
    if agent == None:
        special_df = df.loc[id:id, ['출원번호','요약']]
        # print("special_df:")
        # print(special_df)
        # csv_memory = ConversationBufferMemory(return_messages=True)
        agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"),
                # model,
                special_df, 
                verbose=True,
                reduce_k_below_max_tokens=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True
                )
        # agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                        # verbose=True, memory=csv_memory)

    response = agent.run(user_query)
    print("response:", response)

    return history, agent