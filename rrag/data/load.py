import json
from typing import List


def load_data(data_pth: str, forma: str = "json"):
    if forma == "json":
        with open(data_pth, 'r', encoding='utf-8') as R:
            data = json.load(R)
    elif forma == "jsonl":
        with open(data_pth, 'r', encoding='utf-8') as R:
            data_l = R.readlines()
            data = [json.loads(itm) for itm in data_l]
    return data


def check_data_format(data: List) -> bool:
    try:
        for itm in data:
            q, a, c = itm["question"], itm["answer"], itm["context"]
        return True
    except Exception as e:
        return False
