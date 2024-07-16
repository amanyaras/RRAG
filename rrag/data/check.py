import json
import random


def check_data(file_pth: str):
    with open(file_pth, 'r', encoding="utf-8") as R:
        data = json.load(R)
    random.shuffle(data)
    return data[0]
