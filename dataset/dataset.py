import json


def get_data():
    with open('dataset/data.json') as f:
        data = json.loads(f.read())
        return data

