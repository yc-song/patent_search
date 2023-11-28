import argparse
import os
import re
import json
import time

import tiktoken
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer
from scipy.stats import rankdata
 
def aggregate_chunks(chunk_dicts):
    all_chunks = [x['chunks'] for x in chunk_dicts]
    all_chunks = sum(all_chunks, [])
    return all_chunks

# def mrr(out, labels, input = None): #implement mean reciprocal rank
#     idx_array = rankdata(-out, axis=1, method='min')
#     labels = labels.astype(int)
#     rank = np.take_along_axis(idx_array, labels[:, None], axis=1)
#     return np.sum(1/rank)

# def recall(out, labels, k=4):
#     idx_array = rankdata(-out, axis=1, method='min')
#     labels = labels.astype(int)
#     rank = np.take_along_axis(idx_array, labels[:,None], axis=1)
#     return np.count_nonzero(rank<=k)

def main(args):
    pd.set_option("display.max_columns", 999)
    os.chdir('./')
    # declare tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", use_fast=True, max_length=1024)
    # Document Embeddings
    try: 
        # if index is stored beforehand, load it.
        vectorstore = FAISS.load_local(args.save_dir, embeddings)
    except:
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=300, chunk_overlap=10)
        doc_chunk_dicts=[]
        text_columns = ['요약', '청구항', '기술분야', '배경기술', '해결하려는 과제', '과제의 해결 수단',
            '발명의 효과', '도면의 간단한 설명', '발명을 실시하기 위한 구체적인 내용']
        sample_data = pd.read_csv(args.doc_path)
        for i, row in sample_data.iterrows():
            patent_dict = dict( 
                # 특허 번호 따기
                application_number = str(row['출원번호']), # 출원 번호
                publication_number = str(row['공개번호']), # 공개 번호
                patent_number = str(row['등록번호']), # 등록 번호
                chunks = [],
            )
            
            # text column들을 돌면서, chunking 하기
            chunks = []
            drawing_nums_list = []
            refs_list = []
            for col in text_columns:
                if (col in ['도면의간단한설명', '부호의설명']) or (str(row[col]) == 'nan'): # 도면의 간단한 설명, 부호의 설명은 embed 하지 않음!
                    continue
                curr_section_chunks = text_splitter.create_documents([str(row[col])], [{"application_number": str(row['출원번호'])}])
                chunks.extend(curr_section_chunks)
                
            # patent_dict['chunks'] = list(zip(chunks, zip(refs_list, drawing_nums_list)))
            patent_dict['chunks'] = chunks
            doc_chunk_dicts.append(patent_dict)    
            all_chunks = aggregate_chunks(doc_chunk_dicts)

            # Embed documents
            model_name = "intfloat/multilingual-e5-large"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

            ## https://huggingface.co/intfloat/multilingual-e5-large
            ## this model requires prepending "query: " to texts if the texts are for semantic similarity search task
            vectorstore = FAISS.from_documents(all_chunks, embedding=embeddings)
            if args.save:
                # If you want to save vectorstore, run python main.py --save
                vectorstore.save_local(args.save_path)
    # Load Prior Arts 
    prior_data = pd.read_csv(args.prior_path)
    prior_dict = {} # Key - 쿼리, Value - prior art의 출원번호
    print(prior_data)
    # 딕셔너리 만듦
    # Process each line in the data
    # ToDo: debug
    for line in prior_data.iterrows():
        key, value = line.split(',')
        # Append the value to the list of values for this key
        prior_dict.setdefault(key, []).append(value)
    print(prior_dict)


    for i, row in sample_data.iterrows():
        if int(row['query']) == 1:
            patent_dict = dict( 
                # 특허 번호 따기
                application_number = str(row['출원번호']), # 출원 번호
                publication_number = str(row['공개번호']), # 공개 번호
                patent_number = str(row['등록번호']), # 등록 번호
                abstract = str(row['요약']),
            )
        # patent_dict['abstract'] = text_splitter.create_documents([str(row['요약'])], [{"application_number": str(row['출원번호'])}])
        docs = vectorstore.similarity_search(patent_dict['abstract'], k=10)
        # if gold == 0: 위에 것 그냥 쓰면 됨
        # elig gold == 1: 자기자신을 제외해줌
        # recall, mrr을 구해야 함
            # gold의 가장 높은 ranking을 구함 -> 쭉 append
            # for loop을 다 돌고 나면
                # recall: rank<k인 것들의 개수 / 전체 개수
                # mrr: 1/rank의 평균

    for doc in docs:
        print(doc.metadata['application_number'])
        print(doc.page_content)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_path', type = str, default = './data/patent_data_text_final.csv',
                        help='path to csv file')
    parser.add_argument('--prior_path', type = str, default = './data/prior_arts.csv',
                        help='path to prior arts file')
    parser.add_argument('--save_path', type = str, default = './data',
                        help='path to saving vector embedding file')
    parser.add_argument('--save', action = 'store_true',
                        help='whether to save')
    
    args = parser.parse_args()   
    main(args)




    # if query == 1 and gold == 0:
    #     find@k
    # elif query == 1 and gold == 1:
    #     retrieved_sets = find@(k+1)
    #     retrieved_sets.pop(itself)
    #     retrieved_sets = retrieved_sets[:k]
    