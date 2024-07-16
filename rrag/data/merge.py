import os
import json


def merge_data(file_lst, save_pth: str):
    sv_lst = []
    for itm in file_lst:
        with open(itm, 'r', encoding='utf-8') as R:
            data = json.load(R)
            sv_lst.extend(data)
    with open(save_pth, 'w', encoding='utf-8') as W:
        json.dump(sv_lst, W, ensure_ascii=False, indent=4)
