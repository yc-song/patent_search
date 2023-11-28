from ML.utils import given_information_determine, load_text_index, load_image_index, load_text_and_image_index
from ML.distance_search import topk_search_queryEmb
from ML.clip import embedding
from ML.RAG import textRAG_from_ids

def ML_RAG_result():
    results, revised_results = textRAG_from_ids(df, ids, query, max_rows = 3)

def ML_topk_result(query, image_name, FRONT_ENDSYMBOL):
    """
    
    Return example:
    {"out": [{
                "summary": "본 발명은 쿠션부재의 표면에 착용자의 신체조건에 맞는 보조패드를 부착하여 손목의 접힘각도를 조절하고, 그에 따라 볼 회전력을 증가시킬 수 있게 하고 볼 컨트롤을 보다 자유롭게 구사할 수 있게 한 볼링용 손목보호대의 보조패드에 관한 것이다.",
                "image": "../backend/image/1-1.png"},
            {
                "summary": "본 개시에 따른 기술적 사상은 훈련간 스마트워치 및 훈련 인원의 손목을 동시에 보호할 수 있도록 소정의 영역 에 스마트워치의 시간을 볼 수 있도록 구멍이 형성된 보호대, 상기 보호대의 상부 및 상기 구멍에 인접한 위치에 구비되며, 스마트워치에 표시된 시간을 볼 수 있도록 열고 닫힘을 수행할 수 있도록 뚜껑과 본체를 포함하는 보 호캡 및 복수의 결합부재들 중 적어도 일부가 서로 결합되어 스마트워치 및 손목을 보호할 수 있도록 구비된 결 합부를 포함하는 스마트워치 손목보호대에 관한 것이다.",
                "image": "../backend/image/1-2.png"}]}
    """
    # 텍스트 쿼리나 이미지 중 어떤 정보가 유저로부터 입력되었는지 결정
    textOnly, imageOnly, BothExists, BothNone = given_information_determine(query, image_name, FRONT_ENDSYMBOL) 
    
    # 입력된 정보의 종류별로 faiss index 파일 추출
    if BothNone:
        multi_modal_mode = False
        mode = 'None'
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
        
    query_emb, image_emb = embedding(query, image_name, species = mode)
    distances, indices = topk_search_queryEmb(query_emb, image_emb, textIndex, imageIndex, multi_modal_mode=multi_modal_mode)
    
    return distances, indices