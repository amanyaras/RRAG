from vllm import SamplingParams, LLM


def generate_data(**kwargs):
    question_lst = kwargs["question_lst"]
    model_path = kwargs["model_name_or_path"]
    max_bs = int(kwargs["max_bs"])
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.50,
        max_tokens=1024
    )
    llm = LLM(model_path, tensor_parallel_size=4, trust_remote_code=True,
              gpu_memory_utilization=0.95)
    output_lst = []
    for batch in range(0, len(question_lst), max_bs):
        ans = llm.generate(question_lst[batch:batch+max_bs], sampling_params) if batch+max_bs < len(question_lst) else \
            llm.generate(question_lst[batch:], sampling_params)
        out = [itm.outputs[0].text + "\n" for itm in ans]
        output_lst.extend(out)
    pass
    return output_lst
