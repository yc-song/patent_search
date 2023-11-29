from PIL import Image
# import requests
import torch, os
import torch.nn.functional as F
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
import pandas as pd
import argparse



def embedding(query, image_path, species, model, processor):

    """Find model embedding given content and species.

    Args:
        content (str): Either 'user query' or 'image file path'
            If content == user_query, then directly enter this text to the model to obtain the model embedding
            If content == 'image file path', then load the image using this path, and enter this image to the model to obtain the model embedding.
            
        species (str): specify what the species of this content is. Either 'text' or 'image' or 'both'.
                        If none of them, then raise error.
    """
    
    # text인 경우에 [텍스트1, 텍스트2] 형태로 바꿔야 함.
    if species == 'text' or 'both':
        if type(query) == str:
            query = [query]
       
    # image 불러오기. 만약 받은 이미지가 없으면 샘플 이미지를 사용함.
    # 이래도 텍스트 임베딩에는 변화를 주지 않음. 
    if species == 'text':
        image = Image.open("ML/sample_image.jpeg")
    else:
        image = Image.open(image_path)
    
    inputs = processor(
    text=query,
    images=image, 
    return_tensors="pt", # could also be "pt" 
    padding=True
    )
    outputs = model(**inputs)
    
    # te ie shape: (dataNum, 512)
    te = outputs.text_embeds
    ie = outputs.image_embeds
    
    return te, ie

def embedding_text(query, model):
    
    print("embedding_text. query:", query)
    docs_for_embed = chunk_text(query)
    print("docs_for_emb", docs_for_embed)
    if len(docs_for_embed)>1:
        docs_for_embed = docs_for_embed[:-1]
    
    print(f"docs_for_embed[0]:{docs_for_embed[0]}")
    print(f"docs_for_embed[-1]:{docs_for_embed[-1]}")
    
    outputs = model.encode(docs_for_embed) # shape(lists_size, 512)
    if len(outputs.shape) > 1:
        outputs = outputs.sum(axis = 0)
    # normalize embeddings
    return outputs

def embedding_image(image_path, model):
    if not os.path.isfile(image_path):
        return torch.tensor([1])
    image = Image.open(image_path)
    outputs = model.encode(image) # shape (,512)

    return outputs

def truncate_sentence(sentence, chunk_size):
    # Split the sentence into words
    words = sentence.split()

    # Consider up to the first 1000 words
    truncated_sentence = ' '.join(words[:chunk_size])

    return truncated_sentence

def chunk_text(sentence, chunk_size = 200, overlap = 10):
    start = 0
    end = 0
    result = []
    splitted = sentence.split()
    
    total_words = len(splitted)
    if total_words == 1:
        return splitted
    
    while end < total_words-1:
        end = start + chunk_size
        sentence_list = splitted[start:end]
        result.append(' '.join(sentence_list))
        start = start + chunk_size - overlap        
        
    return result