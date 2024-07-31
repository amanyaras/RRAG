import json
from datasets import load_dataset
from data.mapping import DATA_PATH


if __name__ == "__main__":
    print(len(DATA_PATH))
    print(DATA_PATH[4])
    # wikiQA_gpt
    data = load_dataset(DATA_PATH[4])
    print(data)
    exit()
    print(data["validation"][0]["context"])
    print(data["validation"][0]["answers"])
    print(data["validation"][0]["question"])
    sv_lst = []

    for itm in range(len(data["train"])):
        # if "None" in [data["validation"][itm]["question"], data["validation"][itm]["answers"], data["validation"][itm]["context"]]:
        #     continue

        # print(data["train"][itm]["answer"])
        sv_lst.append({
            "question": data["train"][itm]["question"],
            "answer": data["train"][itm]["answers"]["text"],
            "context": data["train"][itm]["context"]
        })

    with open("/home/zhangyh/rag_dataset/ms_marco.json", 'w', encoding='utf-8') as W:
        json.dump(sv_lst, W, ensure_ascii=False, indent=4)

    print(len(sv_lst))
    pass
