from PIL import Image
# import requests
# import torch



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
