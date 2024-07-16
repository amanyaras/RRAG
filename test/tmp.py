import os
from rrag.data.load import load_data
from rrag.generator import generate_data


if __name__ == "__main__":
    data = load_data("/home/zhangyh/rag_dataset/wikiQA_gpt.json")
    args = {
        "question_lst": ["你是谁？"]*30,
        "model_path": "/home/zhangyh/models/Qwen2-7B-Instruct",
        "max_bs": 512
    }
    result = generate_data(**args)
    print(result[:9])
    print(len(data))
    # print(123)
    pass
    pass
    pass
