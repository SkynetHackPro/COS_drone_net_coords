#!/usr/bin/env python
# coding: utf-8
import json
import os
from random import randrange

import pandas as pd
from tqdm import tqdm

FRAMES_COUNT = 4
RAW_FILE_PATH = 'data_raw/tracks.csv'
RAW_2_FILE_PATH = 'data_raw/tracks_small.csv'

OUT_DIR = 'data'
JSON_DATASET = 'data/dataset.json'


def create_dirs():
    try:
        os.mkdir(OUT_DIR)
    except Exception as e:
        print(e)


def get_frame_tensor_from_row(row):
    return {
        'x': row.x,
        'y': row.y,
        'w': row.w,
        'h': row.h
    }


def make_json_dataset():
    tensor_lists = []

    all_data = pd.read_csv(RAW_FILE_PATH)
    all_data_2 = pd.read_csv(RAW_2_FILE_PATH)

    inq = all_data['id'].unique()[-1] + 1
    all_data_2['id'] += inq

    all_data = pd.concat([all_data, all_data_2])

    track_ids_list = all_data['id'].unique()

    for track_id in tqdm(track_ids_list):
        track_data = all_data.loc[all_data['id'] == track_id]
        tracked_obj = track_data.iloc[0].label
        length = track_data.shape[0]

        for start_index in range(0, length, FRAMES_COUNT):
            data_for_tensor = track_data[start_index: start_index + FRAMES_COUNT]
            if len(data_for_tensor) != FRAMES_COUNT:
                continue
            tensor_data = []

            for i in data_for_tensor.iterrows():
                tensor_data.append(get_frame_tensor_from_row(i[1]))

            data = {
                'label': tracked_obj,
                'data': tensor_data
            }
            tensor_lists.append(data)

    with open(JSON_DATASET, 'w') as f:
        f.write(json.dumps(tensor_lists))


def pick_random_items(items, pick_count):
    picked = []
    for i in range(pick_count):
        index = randrange(0, len(items))
        picked.append(items[index])
        items.pop(index)
    return picked


def _split_dataset(items, train_part, val_part):
    items_count = len(items)
    train_count = round(items_count * train_part)
    val_count = round(items_count * val_part)

    train_dataset = pick_random_items(items, train_count)
    val_dataset = pick_random_items(items, val_count)

    return [train_dataset, val_dataset, items]


def split_dataset():
    with open(JSON_DATASET, 'r') as f:
        data = json.loads(f.read())

        [train, test, val] = _split_dataset(data, 0.6, 0.2)
        with open(f'{OUT_DIR}/train.json', 'w') as t:
            t.write(json.dumps(train))

        with open(f'{OUT_DIR}/test.json', 'w') as t:
            t.write(json.dumps(test))

        with open(f'{OUT_DIR}/val.json', 'w') as t:
            t.write(json.dumps(val))


if __name__ == '__main__':
    create_dirs()

    make_json_dataset()

    split_dataset()
