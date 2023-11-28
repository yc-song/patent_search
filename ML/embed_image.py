from PIL import Image
import torch
import clip
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import re
from tqdm import tqdm
import openpyxl
# https://medium.com/@jeremy-k/unlocking-openai-clip-part-3-optimizing-image-embedding-storage-and-retrieval-pickle-vs-faiss-25d0f02c049d
#Load CLIP

# def extract_number_from_pattern(string):
#     # Regular expression pattern to match 'pdf_xxxx/pxx-xx.png' where xxxx is the number we want
#     pattern = r'pdf_(\d+)/p(\d+)-(\d+)\.png'

#     # Using regular expression to find the number
#     match = re.search(pattern, string)
#     if match:
#         return match.group(1)
#     else:
#         return None
        
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = SentenceTransformer('clip-ViT-B-32')

#Define the emb variable to store embeddings
emb = {}

#read all image names
images = []
for root, dirs, files in os.walk('./image'):
    for file in files:
        if file.endswith('png'):
            images.append(root  + '/'+ file)
previous_image = None
count = 0
print('extract embeddings and store them in the emb variable')
images_list = list()
model = model.to(device)
for img in tqdm(images):
    # images_list.append(img)
    with torch.no_grad():
        image = Image.open(img)
        if previous_image != img: 
            count = 0
            previous_image = img
        else: count += 1
        image_features = model.encode(image)
        if count<10: emb_key = img + '0' + str(count)
        else: emb_key = img + str(count)
        emb[emb_key] = image_features
    torch.save(emb, './data/emb.pt')
#Create Faiss index using FlatL2 type. 512 is the number of dimensions of each vectors
index = faiss.IndexFlatL2(512)
#Convert embeddings and add them to the index
for key in tqdm(emb):
    #Convert to numpy
    #Convert to float32 numpy
    vector = np.float32(emb[key])
    vector = np.expand_dims(vector, 0)

    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)
#Store the index locally
faiss.write_index(index,"./embedding/vector_image.index")
