import argparse
import os
import re
import json
import time

# from typing import List, Tuple
import torch
import tiktoken
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import Tensor
import torch.nn.functional as F
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModel
from scipy.stats import rankdata
 
def aggregate_chunks(chunk_dicts, model_name=""):
    all_chunks = [x['chunks'] for x in chunk_dicts]
    all_chunks = sum(all_chunks, [])
    if model_name == "intfloat/multilingual-e5-large":
        for doc in all_chunks:
            doc.page_content = "query: " + doc.page_content
    return all_chunks

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


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
import numpy as np
def calculate_recall(_docs_with_score, _labels):
    _docs = [x[0].metadata['application_number'] for x in _docs_with_score] # list of application numbers with length same as docs_with_score
    _yn = [1 if x in _labels else 0 for x in _docs] # list of 1,0 with length same as docs_with_score
    _scores = np.array([x[1] for x in _docs_with_score]) # array of shape (length docs_with_score,) of similarity scores
    
    idx_array = rankdata(_scores, method='max')
    idx_array_gold = idx_array[np.where(np.array(_yn)==1)[0]] # only get ranks of indices where application number is in the labels (gold prior arts)
    if len(idx_array_gold) == 0:
        return 0.
    else:
        return len(idx_array_gold) / len(_docs_with_score)
    
def calculate_rr(_docs_with_score, _labels):
    
    _docs = [x[0].metadata['application_number'] for x in _docs_with_score] # list of application numbers with length same as docs_with_score
    _yn = [1 if x in _labels else 0 for x in _docs] # list of 1,0 with length same as docs_with_score
    _scores = np.array([x[1] for x in _docs_with_score]) # array of shape (length docs_with_score,) of similarity scores
    
    idx_array = rankdata(_scores, method='max')
    idx_array_gold = idx_array[np.where(np.array(_yn)==1)[0]] # only get ranks of indices where application number is in the labels (gold prior arts)
    if len(idx_array_gold) == 0:
        return np.nan
    else:
        return 1 / min(idx_array_gold)


 # Embed documents -> FAISS IS NOT WORKING
# model을 돌려서 all_chunks에 대한 임베딩 얻어야 함
# 메타데이터 없는게 문제임. 
# 0. document를  model에 돌려야 함
## intfloat/multilingual-e5-large
## all_chunks에 대해 위 모델에 돌려서 임베딩을 저장
## t7으로 저장 -> 차후에 로드도 가능
# 1. 이는 딕셔너리로 임베딩들을 저장하면 됨. 
## Key - 출원번호, 임베딩 
## e.g. [{"출원번호": 10200110129138, "임베딩": [0.1111, 0.2222, 0.3333]}
# 2. query 도 모델에 돌려야 함
## query에 대해 위 모델에 돌려서 임베딩을 저장
# 3. Score 측정
## out = torch.bmm(document_embedding, query_embedding.transpose(1,0))
# 4. Rank 측정
## rank = np.take_along_axis(-out, labels[:, None], axis=1)
# 5. recall, mrr 측정

def main(args):
    pd.set_option("display.max_columns", 999)
    os.chdir('./')
    # declare tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", use_fast=True, max_length=1024)
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, 
                                                                         chunk_size=500, 
                                                                         chunk_overlap=10)
    doc_chunk_dicts=[]
    doc_embedding_list = []
    text_columns = ['요약', '청구항', '기술분야', '배경기술', '해결하려는 과제', '과제의 해결 수단',
        '발명의 효과', '도면의 간단한 설명', '발명을 실시하기 위한 구체적인 내용']
    sample_data = pd.read_csv(args.doc_path, dtype=str)
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
        
        patent_dict['chunks'] = chunks
        doc_chunk_dicts.append(patent_dict)    
        
        docs_for_embed = []
        for doc in chunks:
            doc.page_content = "query: " + doc.page_content
            docs_for_embed.append(doc.page_content)
        docs_for_embed = torch.stack(docs_for_embed, axis=0)
        batch_dict = tokenizer(docs_for_embed, max_length=512, padding=True, truncation=True, return_tensors='pt')

        # pass through outputs
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # save into dictionary
        embed_dict_for_doc = []
        for i, doc in enumerate(chunks):
            embed_dict_for_doc.append({"출원번호": patent_dict['application_number'], 
                                       "임베딩": embeddings[i]})
            
        doc_embedding_list.extend(embed_dict_for_doc)
    
    
#     # Document Embeddings
#     try: 
        
#         # Embed documents
#         model_name = "intfloat/multilingual-e5-large"
#         model_kwargs = {'device': 'cpu'}
#         encode_kwargs = {'normalize_embeddings': True}
#         embeddings = HuggingFaceEmbeddings(
#             model_name=model_name,
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs,
#         )
#         # if index is stored beforehand, load it.
#         vectorstore = FAISS.load_local(args.save_path, embeddings)
#         sample_data = pd.read_csv(args.doc_path, dtype=str)
#     except Exception as e:
#         text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, 
#                                                                          chunk_size=300, 
#                                                                          chunk_overlap=10)
#         doc_chunk_dicts=[]
#         text_columns = ['요약', '청구항', '기술분야', '배경기술', '해결하려는 과제', '과제의 해결 수단',
#             '발명의 효과', '도면의 간단한 설명', '발명을 실시하기 위한 구체적인 내용']
#         sample_data = pd.read_csv(args.doc_path, dtype=str)
#         for i, row in sample_data.iterrows():
#             patent_dict = dict( 
#                 # 특허 번호 따기
#                 application_number = str(row['출원번호']), # 출원 번호
#                 publication_number = str(row['공개번호']), # 공개 번호
#                 patent_number = str(row['등록번호']), # 등록 번호
#                 chunks = [],
#             )
            
#             # text column들을 돌면서, chunking 하기
#             chunks = []
#             drawing_nums_list = []
#             refs_list = []
#             for col in text_columns:
#                 if (col in ['도면의간단한설명', '부호의설명']) or (str(row[col]) == 'nan'): # 도면의 간단한 설명, 부호의 설명은 embed 하지 않음!
#                     continue
#                 curr_section_chunks = text_splitter.create_documents([str(row[col])], [{"application_number": str(row['출원번호'])}])
#                 chunks.extend(curr_section_chunks)
                
#             # Embed documents
#             model_name = "intfloat/multilingual-e5-large"
#             model_kwargs = {'device': 'cpu'}
#             encode_kwargs = {'normalize_embeddings': True}
#             embeddings = HuggingFaceEmbeddings(
#                 model_name=model_name,
#                 model_kwargs=model_kwargs,
#                 encode_kwargs=encode_kwargs,
#             )
            
#             # patent_dict['chunks'] = list(zip(chunks, zip(refs_list, drawing_nums_list)))
#             patent_dict['chunks'] = chunks
#             doc_chunk_dicts.append(patent_dict)    
#         all_chunks = aggregate_chunks(doc_chunk_dicts, model_name)

#         ## https://huggingface.co/intfloat/multilingual-e5-large
#         ## this model requires prepending "query: " to texts if the texts are for semantic similarity search task
#         vectorstore = FAISS.from_documents(all_chunks, embedding=embeddings)
#         if args.save:
#             # If you want to save vectorstore, run python main.py --save
#             vectorstore.save_local(args.save_path)
#     # Load Prior Arts 
#     prior_data = pd.read_csv(args.prior_path,names=['source','target'], dtype=str)
#     prior_dict = {} # Key - 쿼리, Value - prior art의 출원번호
#     # 딕셔너리 만듦
#     # Process each line in the data
#     for i, line in prior_data.iterrows():
#         key, value = line['source'], line['target']
#         # Append the value to the list of values for this key
#         prior_dict.setdefault(key, []).append(value)
#     # print(prior_dict)

#     recalls = []
#     rrs = []
#     for i, row in sample_data.iterrows():
#         if (int(row['query']) == 1) and (int(row['labelled_yn']) == 1):
            
#             patent_dict = dict( 
#                 # 특허 번호 따기
#                 application_number = str(row['출원번호']), # 출원 번호
#                 publication_number = str(row['공개번호']), # 공개 번호
#                 patent_number = str(row['등록번호']), # 등록 번호
#                 abstract = str(row['요약']),
#             )
#             # patent_dict['abstract'] = text_splitter.create_documents([str(row['요약'])], [{"application_number": str(row['출원번호'])}])
#             docs = vectorstore.similarity_search_with_score(patent_dict['abstract'], k=20)
            
#             # 자기 자신을 제외 시켜줌 
#             docs = [x for x in docs if str(x[0].metadata['application_number']) != str(patent_dict['application_number'])]
            
#             recall = calculate_recall(docs, prior_dict[patent_dict['application_number']])
#             recalls.append(recall)
            
#             rr = calculate_rr(docs, prior_dict[patent_dict['application_number']])
#             rrs.append(rr)
            
#     average_recall = np.mean(np.array(recalls))
#     mrr = np.nanmean(np.array(rrs))
    
#     print(f"average recall: {average_recall}")
#     print(f"mrr: {mrr}")
    
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--doc_path', type = str, default = './data/patent_data_text_final.csv',
#                         help='path to csv file')
#     parser.add_argument('--prior_path', type = str, default = './data/prior_arts.csv',
#                         help='path to prior arts file')
#     parser.add_argument('--save_path', type = str, default = './data',
#                         help='path to saving vector embedding file')
#     parser.add_argument('--save', action = 'store_true',
#                         help='whether to save')
    
#     args = parser.parse_args()   
#     main(args)




#     # if query == 1 and gold == 0:
#     #     find@k
#     # elif query == 1 and gold == 1:
#     #     retrieved_sets = find@(k+1)
#     #     retrieved_sets.pop(itself)
#     #     retrieved_sets = retrieved_sets[:k]
    