import torch


def dict_to_list(x):
    return [
        x['x'],
        x['y'],
        x['w'],
        x['h']
    ]


def data_to_tensor(data):
    return torch.tensor([[dict_to_list(i) for i in data]], dtype=torch.float)


def load_model():
    model = torch.load('model/model.pt', map_location=torch.device('cpu'))
    model.eval()
    return model


def predict(data):
    t = data_to_tensor(data)
    model = load_model()
    pred = model(t)
    return {'drone': '{:.10f}'.format(float(torch.nn.functional.softmax(pred, dim=1).data.cpu().numpy()[0][1]))}
