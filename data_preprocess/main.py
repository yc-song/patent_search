from PIL import Image
import torch
import clip
import numpy as np
import faiss
import os
import transformers
from sentence_transformers import SentenceTransformer
index_image  = faiss.read_index("./data_preprocess/vector_image.index")
index_text = faiss.read_index("./data_preprocess/vector_text.index")
#Tokenize the prompt to search using CLIP
query = '\
본 발명은 이중관 구조를 갖는 온수 분배관 내에 열선이 구비된 보일러 가열장치에 관한 것\
'
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
text_features = model.encode(query)
text_features = np.expand_dims(text_features, 0)

#Preprocess the tensor
text_np = np.float32(text_features)
faiss.normalize_L2(text_np)

#Search the top 5 images
probs, indices = index_image.search(text_np, 5)
# probs, indices = index_text.search(text_np, 1)

print('probs',probs)
print('indice' ,indices)