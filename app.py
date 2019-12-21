import json
from flask import Response
from flask import Flask
from flask import request

from dataset import dataset
from model import predict

app = Flask(__name__)


@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    return Response(json.dumps(dataset.get_data()), mimetype='application/json')


@app.route('/api/predict', methods=['POST'])
def test():
    payload = request.get_json()
    model_results = predict.predict(payload['data'])
    return Response(json.dumps(model_results), mimetype='application/json')
