import json


def alpaca2fs(meta_data_lst):
    sv_lst = []
    for i, itm in enumerate(meta_data_lst):
        sv_lst.append({
            "id": str(i),
            "conversations": [
                {
                    "from": "human",
                    "value": itm["instruction"] + itm["input"]
                },
                {
                    "from": "gpt",
                    "value": itm["output"]
                }
            ]
        })
    return sv_lst


def fs2alpaca(meta_data_lst):
    sv_lst = []
    for idx, itm in enumerate(meta_data_lst):
        sv_lst.append({
            "instruction": itm["conversations"][0]["value"],
            "input": "",
            "output": itm["conversations"][1]["value"]
        })
    return sv_lst
