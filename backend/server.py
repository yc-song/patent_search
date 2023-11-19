##########################################################
# to run: FLASK_APP=server.py flask run
##########################################################
import json
from flask import Flask, request

app = Flask(__name__)


@app.route("/api/data")
def data():
    image_name = request.args.get("image_name")
    query = request.args.get("query")
    print(image_name)
    print(query)
    return {"out": [{"summary": "본 발명은 쿠션부재의 표면에 착용자의 신체조건에 맞는 보조패드를 부착하여 손목의 접힘각도를 조절하고, 그에 따라 볼 회전력을 증가시킬 수 있게 하고 볼 컨트롤을 보다 자유롭게 구사할 수 있게 한 볼링용 손목보호대의 보조패드에 관한 것이다.",
                    "image": "../backend/image/1-1.png"},
                    {"summary": "본 개시에 따른 기술적 사상은 훈련간 스마트워치 및 훈련 인원의 손목을 동시에 보호할 수 있도록 소정의 영역 에 스마트워치의 시간을 볼 수 있도록 구멍이 형성된 보호대, 상기 보호대의 상부 및 상기 구멍에 인접한 위치에 구비되며, 스마트워치에 표시된 시간을 볼 수 있도록 열고 닫힘을 수행할 수 있도록 뚜껑과 본체를 포함하는 보 호캡 및 복수의 결합부재들 중 적어도 일부가 서로 결합되어 스마트워치 및 손목을 보호할 수 있도록 구비된 결 합부를 포함하는 스마트워치 손목보호대에 관한 것이다.",
                     "image": "../backend/image/1-2.png"}]}
