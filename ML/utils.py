import faiss

def load_text_index():
    textIndex = faiss.read_index("./data_preprocess/vector_text.index")
    return textIndex

def load_image_index():
    imageIndex = faiss.read_index("./data_preprocess/vector_image.index")
    return imageIndex

def load_text_and_image_index():
    t = load_text_index()
    i = load_image_index()
    return {"text": t,
            "image": i}
    
def given_information_determine(query, image_name, FRONT_ENDSYMBOL):
    textOnly, imageOnly, BothExists, BothNone = False, False, False, False
    
    if query == FRONT_ENDSYMBOL and image_name == FRONT_ENDSYMBOL:
        BothNone = True
    elif query != FRONT_ENDSYMBOL and image_name == FRONT_ENDSYMBOL:
        textOnly = True
    elif query == FRONT_ENDSYMBOL and image_name != FRONT_ENDSYMBOL:
        imageOnly = True
    else:
        BothExists = True
    
    return textOnly, imageOnly, BothExists, BothNone
