# Create 'keys.json' file in ML folder consists of following three lines:
# {
#     "OpenAI_API_KEY":"your openai api key"
# }

import os, json
import pandas as pd
from langchain_experimental.agents.agent_toolkits.csv.base import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType

with open("keys.json", 'r') as f:
    keys = json.load(f)
OPENAI_API_KEY = keys["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def textRAG_from_ids(df, ids, user_query, max_rows = 3):
    ids = list(set(ids))
    minimized_df = df[['출원번호','요약']].iloc[ids].reset_index()
    # print("minimized_df:")
    # print(minimized_df)
    max_rows = max_rows
    chunks = list(chunk_data_frame(minimized_df, max_rows))
    results = []

    prompt_developers_template = """
    너의 데이터베이스는 특허 파일이야.
    아래와 관련 있는 특허들을 찾아줘:
    {user_query}

    아래의 내용을 반드시 포함하여 답변해줘:
    1. 출원번호
    2. 왜 관련된 특허인지에 대한 상세한 이유
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
        
        revised_result = revise_result(results)
        return results, revised_result


def chunk_data_frame(df, chunk_size):

    num_chunks = len(df) // chunk_size + (len(df) % chunk_size > 0)

    for i in range(num_chunks):

        yield df[i * chunk_size:(i + 1) * chunk_size]
        
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