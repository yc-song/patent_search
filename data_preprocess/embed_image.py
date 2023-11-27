from PIL import Image
import torch
import clip
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import os
# https://medium.com/@jeremy-k/unlocking-openai-clip-part-3-optimizing-image-embedding-storage-and-retrieval-pickle-vs-faiss-25d0f02c049d
#Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('clip-ViT-B-32')

#Define the emb variable to store embeddings
emb = {}

#read all image names
images = []
for root, dirs, files in os.walk('./data_preprocess/sample_imges'):
    for file in files:
        if file.endswith('png'):
            images.append(root  + '/'+ file)

#Extract embeddings and store them in the emb variable
for img in images:
    with torch.no_grad():
        image = Image.open(img)
        image_features = model.encode(image)
        emb[img] = image_features

#Create Faiss index using FlatL2 type. 512 is the number of dimensions of each vectors
index = faiss.IndexFlatL2(512)
#Convert embeddings and add them to the index
for key in emb:
    #Convert to numpy
    #Convert to float32 numpy
    vector = np.float32(emb[key])
    vector = np.expand_dims(vector, 0)

    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

#Store the index locally
faiss.write_index(index,"./data_preprocess/vector_image.index")