import json

import torch
from torch.utils.data import Dataset


def dict_to_list(x):
    return [
        x['x'],
        x['y'],
        x['w'],
        x['h']
    ]


class JSONTrackDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.transform = transform
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def get_imbalance_weights(self):
        drones_count = 0
        other_count = 0
        for i in self.data:
            if i['label'] == 'drone':
                drones_count += 1
            else:
                other_count += 1

        return torch.tensor([drones_count / other_count, drones_count / drones_count], dtype=torch.float)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.transform:
            for i in self.transform:
                item['data'] = i(item['data'], item['copy'] if 'copy' in item else False)

        data = [dict_to_list(i) for i in item['data']]

        label_tensor = torch.tensor([1, 0])
        if item['label'] == 'drone':
            label_tensor = torch.tensor([0, 1])
        result = {
            'label': label_tensor.to(torch.long),
            'data': torch.tensor([data], dtype=torch.float)
        }

        return result
