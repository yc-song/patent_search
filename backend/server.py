##########################################################
# to run: python server.py
##########################################################
import json
import openai
from flask import Flask, render_template, request, redirect, session, jsonify
import os.path
from ML.ml_main import ML_result
app = Flask(__name__)
openai.api_key = "api here"

FRONT_ENDSYMBOL = '0'

@app.route("/api/data", methods=['GET'])
def data():
    image_name = request.args.get("image_name")
    query = request.args.get…