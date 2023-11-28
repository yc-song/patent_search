# Get embedding from CLIP

import os
import re
import copy
import json
import time
import clip
import tiktoken
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import faiss
from sentence_transformers import SentenceTransformer
import transformers
pd.set_option("display.max_columns", 999)
os.chdir('./')

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
# Download the dataset
patent_chunk_dicts=[]
sample_data = pd.read_csv("./text/debug.csv")
text_columns = ['청구항', '발명의 명칭', '기술분야', '배경기술','선행기술문헌',	'발명의 내용'\
	,'해결하려는 과제',	'과제의 해결 수단',	'발명의 효과', '도면의 간단한 설명', '발명을 실시하기 위한 구체적인 내용', '요약']

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=300, chunk_overlap=10)
for i, row in sample_data.iterrows():
        patent_dict = dict( 
            # 특허 번호 따기
            application_number = str(row['출원번호']), # 출원 번호
            publication_number = str(row['공개번호']), # 공개 번호
            patent_number = str(row['등록번호']), # 등록 번호
            chunks = [],
        )
        print(f"patent {patent_dict['application_number']} 's chunk sizes")
        
        # text column들을 돌면서, chunking 하기
        chunks = []
        drawing_nums_list = []
        refs_list = []
        for col in text_columns:
            if (col in ['도면의간단한설명', '부호의설명']) or (str(row[col]) == 'nan'): # 도면의 간단한 설명, 부호의 설명은 embed 하지 않음!
                continue
            # curr_section_chunks = text_splitter.create_documents([str(row[col])], [{"application_number": str(row['출원번호'])}])
            chunks.append(str(row[col]))
            
            
            # # chunk 별로 reference 발생한 부호 찾기
            # for chnk in curr_section_chunks:
            #     refs = extract_ref_from_text(chnk)
            #     refs_list.append(refs)
                
            #     #  부호가 있었으면 drawing number 로 한번 또 치환하기
            #     if len(refs) > 0: # if not empty, convert found references to drawing numbers
            #         drawing_nums = convert_refs_to_drawing_num(refs, num2drawing_dicts[str(row.id)]['num2drawing'])
            #     else:
            #         drawing_nums = []
            #     drawing_nums_list.append(drawing_nums)
                    
            
        # patent_dict['chunks'] = list(zip(chunks, zip(refs_list, drawing_nums_list)))
        patent_dict['chunks'] = chunks
        print()
        patent_chunk_dicts.append(patent_dict)    
    # sample: 1020190046815
number_chunks = [x['application_number'] for x in patent_chunk_dicts]
all_chunks = [x['chunks'] for x in patent_chunk_dicts]
len_chunks = [len(x['chunks']) for x in patent_chunk_dicts]
previous_id = None

# number_chunks = [index for index, count in zip(number_chunks, len_chunks) for _ in range(count)]
number_chunks_new = []
global_count = 0
for index, count in zip(number_chunks, len_chunks):
    for i in range(count):
        global_count+=1
        # Add suffixes to the first three occurrences
        if i<10:
            number_chunks_new.append(index + '0' + str(i))
        else:
            number_chunks_new.append(index + str(i))
        # print(number_chunks, len_chunks)
    # if i<2:
        # number_chunks_previous[i] = []
        # for j in range(len_chunks[i]):
        #     local_indices = [f"{i:02}" for i in range(len_chunks[i]+1)]
        #     for index in local_indices:
        #         number_chunks_previous[i].append(number_chunks[i] + index)

# all_chunks = [x['chunks'] for x in patent_chunk_dicts if x['application_number'] != '1020190046815']
all_chunks = sum(all_chunks, [])
#Load CLIP
# device = "cuda" if torch.cuda.is_available() else "cpu"
# text_config = transformers.CLIPTextConfig(max_position_embeddings=2000)
# config_vision = transformers.CLIPVisionConfig()
# configuration = transformers.CLIPConfig.from_text_vision_configs(text_config, config_vision)
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
#Define the emb variable to store embeddings
emb = {}

#read all image names
#Extract embeddings and store them in the emb variable
for i, text in tqdm(enumerate(all_chunks)):
    with torch.no_grad():
        # text = tokenizer([text], max_length=1000, truncation=True, padding="max_length", return_tensors="np")
        text_features = model.encode(text)
        emb_key = number_chunks[i]
        emb[emb_key] = text_features

#Create Faiss index using FlatL2 type. 512 is the number of dimensions of each vectors
index = faiss.IndexFlatL2(512)
#Convert embeddings and add them to the index
for key in emb:
    #Convert to numpy
    #Convert to float32 numpy
    vector = np.float32(emb[key])
    #Normalize vector: important to avoid wrong results when searching
    vector = np.expand_dims(vector, 0)
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

#Store the index locally
faiss.write_index(index,"./data_preprocess/vector_text.index")
