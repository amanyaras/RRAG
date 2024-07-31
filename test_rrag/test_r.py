from rrag.data.load import load_data, check_data_format
from rrag.eval.evaluator import cal_rouge
from rrag.generator.generate import generate_data
from rrag.eval.template import RAG_TEMPLATE


def get_rouge(**kwargs):
    """
    data format must be rrag format
    :param kwargs:
    :return:
    """
    data_path = kwargs["data_pth"]
    data = load_data(data_pth=data_path)
    assert check_data_format(data)

    target = [itm["answer"] for itm in data]

    question_lst = [RAG_TEMPLATE.format(itm["context"], itm["question"]) for itm in data]

    kwargs["question_lst"] = question_lst
    result = generate_data(**kwargs)
    score = cal_rouge(target, result)

    return score
