from vllm import SamplingParams, LLM


def generate_data(**kwargs):
    question_lst = kwargs["question_lst"]
    model_path = kwargs["model_path"]
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.50,
        max_tokens=1024
    )
    llm = LLM(model_path, tensor_parallel_size=4, trust_remote_code=True,
              gpu_memory_utilization=0.95)
    output_lst = []
    for batch in range(0, len(question_lst), 512):
        ans = llm.generate(question_lst[batch:batch+512], sampling_params) if batch+512 < len(question_lst) else \
            llm.generate(question_lst[batch:], sampling_params)
        out = [itm.outputs[0].text + "\n" for itm in ans]
        output_lst.extend(out)
    pass
    return output_lst
