import sys
sys.path.insert(0, "/home/zhangyh/projs/rrag")
import os
from rrag.data.load import load_data
from rrag.generator import generate_data
from rrag.argument.parser import get_infer_args
from typing import Optional, Dict, Any, List
from rrag.eval.evaluator import cal_rouge
from test_r import get_rouge

# def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
#     callbacks.append(LogCallback())
#     model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)


if __name__ == "__main__":
    # data = load_data("/home/zhangyh/rag_dataset/wikiQA_gpt.json")
    # args = {
    #     "question_lst": ["你是谁？"]*30,
    #     "model_path": "/home/zhangyh/models/Qwen2-7B-Instruct",
    #     "max_bs": 512
    # }
    # result = generate_data(**args)
    model_args, data_args, finetuning_args, generating_args = get_infer_args()
    print(model_args)
    ans = get_rouge()

    # print(result[:9])
    # print(len(data))
    # print(123)
    pass
    pass
    pass
