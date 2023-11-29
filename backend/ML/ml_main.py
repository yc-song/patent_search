from backend.ML.utils import given_information_determine, load_text_index, load_image_index, load_text_and_image_index
from backend.ML.distance_search import topk_search_queryEmb
from backend.ML.clip_emb import embedding, embedding_text, embedding_image
# from RAG import textRAG_from_ids

# def ML_RAG_result():
    # results, revised_results = textRAG_from_ids(df, ids, query, max_rows = 3)

def ML_topk_result(query, image_name, FRONT_ENDSYMBOL, model, processor = None, koclip = False):
    """
    adsf
    """
    # 텍스트 쿼리나 이미지 중 어떤 정보가 유저로부터 입력되었는지 결정
    textOnly, imageOnly, BothExists, BothNone = given_information_determine(query, image_name, FRONT_ENDSYMBOL) 
    print("result:" , textOnly, imageOnly, BothExists, BothNone)
    # 입력된 정보의 종류별로 faiss index 파일 추출
    if BothNone:
        multi_modal_mode = False
        mode = 'None'
        print("BothNone")
        return {"out": [{"summary": "아무런 정보도 입력되지 않았습니다."}]}
    elif textOnly:
        multi_modal_mode = False
        textIndex, imageIndex = load_text_index(), None
        mode = 'text'
    elif imageOnly:
        multi_modal_mode = False
        textIndex, imageIndex = None, load_image_index()
        mode = 'image'
    else:
        multi_modal_mode = True
        textIndex, imageIndex = load_text_and_image_index()
        mode = 'both'
        
    if koclip:
        query_emb, image_emb = embedding(query, image_name, species = mode, model = model, processor = processor)
    else:
        query_emb = embedding_text(query, model[0]).reshape(1, -1)
        image_emb = embedding_image(image_name, model[1]).reshape(1, -1)
    print(f"query_emb shape:{query_emb.shape}")
    print(f"image_emb shape:{image_emb.shape}")
    
    distances, indices = topk_search_queryEmb(query_emb, image_emb, textIndex, imageIndex, multi_modal_mode=multi_modal_mode)

    
    return distances, indices