# from PIL import Image
# import torch
# import clip
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import os
# import pandas as pd
# import re
# from tqdm import tqdm
# import openpyxl
# # https://medium.com/@jeremy-k/unlocking-openai-clip-part-3-optimizing-image-embedding-storage-and-retrieval-pickle-vs-faiss-25d0f02c049d
# #Load CLIP

# def extract_number_from_pattern(string):
#     # Regular expression pattern to match 'pdf_xxxx/pxx-xx.png' where xxxx is the number we want
#     pattern = r'pdf_(\d+)/p(\d+)-(\d+)\.png'

#     # Using regular expression to find the number
#     match = re.search(pattern, string)
#     if match:
#         return match.group(1)
#     else:
#         return None
        
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# model = SentenceTransformer('clip-ViT-B-32')

# #Define the emb variable to store embeddings
# emb = {}

# #read all image names
# images = []
# for root, dirs, files in os.walk('./image'):
#     for file in files:
#         if file.endswith('png'):
#             images.append(root  + '/'+ file)
# previous_image = None
# count = 0
# print('extract embeddings and store them in the emb variable')
# images_list = list()
# model = model.to(device)
# for img in tqdm(images):
#     # images_list.append(img)
#     with torch.no_grad():
#         image = Image.open(img)
#         if previous_image != img: 
#             count = 0
#             previous_image = img
#         else: count += 1
#         image_features = model.encode(image)
#         if count<10: emb_key = img + '0' + str(count)
#         else: emb_key = img + str(count)
#         emb[emb_key] = image_features
#     torch.save(emb, './data/emb.pt')
# #Create Faiss index using FlatL2 type. 512 is the number of dimensions of each vectors
# index = faiss.IndexFlatL2(512)
# #Convert embeddings and add them to the index
# for key in tqdm(emb):
#     #Convert to numpy
#     #Convert to float32 numpy
#     vector = np.float32(emb[key])
#     vector = np.expand_dims(vector, 0)

#     #Normalize vector: important to avoid wrong results when searching
#     faiss.normalize_L2(vector)
#     #Add to index
#     index.add(vector)
# #Store the index locally
# faiss.write_index(index,"./embedding/vector_image.index")


from PIL import Image
import torch
import clip
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = SentenceTransformer('clip-ViT-B-32')

# Define the emb variable to store embeddings
emb = {}

# Read all image names
images = []
for root, dirs, files in os.walk('./image'):
    for file in files:
        if file.endswith('png'):
            images.append(os.path.join(root, file))
# print(images)
# Define a custom dataset for loading images
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        # image = self.transform(image)
        return image, image_path

def custom_collate(batch):
    images, image_paths = zip(*batch)
    # images = torch.stack(images, dim=0)
    return images, image_paths

# Create DataLoader for parallel processing
batch_size = 16  # Adjust the batch size based on your available memory
image_dataset = ImageDataset(images)
data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)

print('Extract embeddings and store them in the emb variable')

model = model.to(device)
for images_batch, image_paths_batch in tqdm(data_loader):
    # print(images_batch.tolist())
    with torch.no_grad():
        # print(model.encode(images_batch))
        # image_features_batch = torch.Tensor(model.encode(images_batch)).to("cpu").numpy()
        image_features_batch = model.encode(images_batch)

    patent_id = ""
    prev_patent_id = ""
    count = 0
    for i, image_path in enumerate(image_paths_batch):
        patent_id = image_path.split('/')[-2]
        if patent_id == prev_patent_id:
            count += 1
        else:
            count = 0
        prev_patent_id = patent_id
        
        if count < 10:
            emb_key = patent_id + '0' + str(count)
        else:
            emb_key = patent_id + str(count)
        print(emb_key)
        emb[emb_key] = image_features_batch[i]

# Save embeddings to file
torch.save(emb, './data/emb.pt')

# Create Faiss index using FlatL2 type. 512 is the number of dimensions of each vector
index = faiss.IndexFlatL2(512)

# Convert embeddings and add them to the index in batches
emb_vectors = np.vstack(list(emb.values()))
emb_vectors = np.float32(emb_vectors)
faiss.normalize_L2(emb_vectors)

for i in tqdm(range(0, len(emb_vectors), batch_size)):
    index.add(emb_vectors[i:i + batch_size])

# Store the index locally
faiss.write_index(index, "./embedding/vector_image.index")
