import json


def load_data(data_pth: str, forma: str = "json"):
    if forma == "json":
        with open(data_pth, 'r', encoding='utf-8') as R:
            data = json.load(R)
    elif forma == "jsonl":
        with open(data_pth, 'r', encoding='utf-8') as R:
            data_l = R.readlines()
            data = [json.loads(itm) for itm in data_l]
    return data
