##########################################################
# to run: python server.py
##########################################################
import json
import openai
from flask import Flask, render_template, request, redirect, session, jsonify
import os.path

app = Flask(__name__)
openai.api_key = "api-here"

FRONT_ENDSYMBOL = '0'


@app.route("/api/data", methods=['GET'])
def data():
    image_name = request.args.get("image_name")
    query = request.args.get("query")
    if os.path.isfile(image_name):
        print(image_name)
        print(query)

    out = ML_result(query, image_name, FRONT_ENDSYMBOL)

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
    request_data = json.loads(request.json)
    print(request_data.get("number"))
    print(request_data.get("messages"))
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=request_data.get("messages")
    )

    data = {
        "response": response.choices[0]["message"]["content"]
    }

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
