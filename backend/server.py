##########################################################
# to run: python server.py
##########################################################
import json, random
import openai
from flask import Flask, render_template, request, redirect, session, jsonify
import pandas as pd
import os.path
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel

from ML.RAG import textRAG_from_ids
from ML.ml_main import ML_topk_result

app = Flask(__name__)
openai.api_key = "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"
global dataframe
global ids
global distances
global indices
global deunglok_to_dfidx
global chulwon_to_dfidx
def from_deunglok_to_dfidx(dataframe):
    out = {}
    deung = dataframe['등록번호']
    for i, d in enumerate(deung):
        try:
            out[str(int(d))] = str(i)
        except:
            continue
    return out

def from_chulwon_to_dfidx(dataframe):
    out = {}
    chulwon = dataframe['출원번호']
    for i, d in enumerate(chulwon):
        try:
            out[str(int(d))] = str(i)
        except:
            continue
    return out

dataframe = pd.read_csv('data_preprocess/extracted_data_formatted.csv')
distances, indices = None, None
print("model load start.")
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
image_model = SentenceTransformer('clip-ViT-B-32')
print("model load end.")
deunglok_to_dfidx = from_deunglok_to_dfidx(dataframe)
chulwon_to_dfidx = from_chulwon_to_dfidx(dataframe)


@app.route("/api/data", methods=['GET'])
def data():
    global distances
    global indices
    global deunglok_to_dfidx
    global chulwon_to_dfidx
    
    image_name = request.args.get("image_name")
    query = request.args.get("query")
    
    FRONT_ENDSYMBOL = '0'
    # model embedding을 이용하여 top-k candidates 뽑기
    print("----------topk search start----------")
    distances, indices = ML_topk_result(query, image_name, FRONT_ENDSYMBOL, (text_model, image_model), koclip = False)
    print("----------topk search complete----------")
    distances = distances[0]
    indices = indices[0]
    if os.path.isfile(image_name):
        print(image_name)
        print(query)
    
    out = {"out":[]}
    print("curr path:", os.getcwd())
    for i in indices:
        if judge_id_species(i) == 'd':
            df_idx = deunglok_to_dfidx[str(i)]
        elif judge_id_species(i) == 'c':
            df_idx = chulwon_to_dfidx.get(str(i)[:-2], random.choice(range(500)))
        chul = dataframe['출원번호'].iloc[int(df_idx)]
        
        # 요약 관련
        summary = dataframe['요약'].iloc[int(df_idx)]
        summary_lastdot = summary.rfind('.')
        summary = summary[:summary_lastdot]
        if len(summary.split()) < 20:
            continue

        # 이미지 관련
        # backend/images/sample 폴더 안에 여러 이미지들을 넣어둬야 함.
        # bacnend/images/출원번호/이미지들... 구조로 이미지를 저장해둬야 함.
        image_path = os.path.join("backend", "image", str(chul))
        selected_image = None
        
        if os.path.exists(image_path):
            image_list = os.listdir(image_path)
            if len(image_list) != 0:
                for i in image_list:
                    if '1' in i:
                        selected_image = i
                        break
        if selected_image == None:
            image_path = os.path.join("backend", "image", "sample")
            image_list = os.listdir(image_path)
            selected_image = random.choice(image_list)
        selected_image_path = os.path.join("..",image_path, selected_image)        
        # print("selected_image_path:", selected_image_path)    
        out["out"].append({
            "summary":summary,
            "image": selected_image_path,
            "chulwon_num":str(chul)
        })
    print("search output calculated.")
    return out


@app.route("/api/chat", methods=['POST'])
def my_api():
    """
    API route to generate response based on dialogue history and user input text and return

    Parameters:
        request (JSON): the input from the user.
    Returns:
        JSON: A JSON object containing the input text and the predicted tags.
    """
    global ids
    global distances
    global indices
    global dataframe
    print(indices)
    if indices == None:
        return { "response": "Chatbot을 사용하기 이전에, Search 탭에서 먼저 원하는 쿼리를 입력해주세요. Search 탭에서의 검색이 끝나면 Chatbot을 사용할 수 있습니다. Chatbot은 Search된 특허 후보를 바탕으로 조금 더 자세한 대화를 제공합니다." }
    ids = [i+1 for i in range(len(indices))]
    request_data = json.loads(request.json)
    print(request_data.get("number"))
    our_id = ids[request_data.get("number")]
    # print(request_data.get("messages"))
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=request_data.get("messages")
    # )

    # data = {
    #     "response": response.choices[0]["message"]["content"] # value is just string.
    # }

    result = textRAG_from_ids(dataframe, [our_id], user_query = request_data)[0] # str
    data = {
        "response": result
    }
    return jsonify(data)

def judge_id_species(i):
    if str(i).endswith('0000'):
        return 'd'
    else:
        return 'c'

if __name__ == "__main__":
    app.run(debug=True)
