from rrag.data.load import load_data
from rrag.eval.evaluator import cal_rouge
from rrag.generator.generate import generate_data


def get_rouge(**kwargs):
    data_path = kwargs["data_pth"]
    data = load_data(data_pth=data_path)
    target = [itm["answer"] for itm in data]
    question_lst = [itm["question"] for itm in data]
    kwargs["question_lst"] = question_lst
    result = generate_data(**kwargs)
    score = cal_rouge(target, result)
    return score
