import json
from typing import List


def save_data(save_pth: str, data: List):
    with open(save_pth, 'w', encoding='utf-8') as W:
        json.dump(data, W, ensure_ascii=False, indent=4)
