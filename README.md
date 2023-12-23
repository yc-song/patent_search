# Multimodal Multiview Patent Search System
- Team 2 (SSSCI)
- Members: Hye Ryung Son, Mooho Song, Jong Hyun Song, Moonwon Choi, Emad Ismael

## Directory structure
- archive: codes that were developed and tried but not included in the final MVP
- backend: codes for backend
- data_collection: codes for data collection (crawling images, parsing texts from texts files downloaded from kipris, and extracting gold prior arts of downloaded patents for further data collection.)
- embedding: codes to embed text and image
- frontend: codes for frontend
- modeling: codes to measure performance metrics for retrieval method

## Setup
Each directory has its own requirements.txt file. 

## Run
### Backend
In the backend directory,
```
python server.py
```
### Frontend
In the frontend directory,
```
streamlit run search.py
```