import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["export VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import torch


import multiprocessing

from rrag.argument.parser import get_infer_args
# from rrag.utils.
from test_rrag.test_r import get_rouge


# def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
#     callbacks.append(LogCallback())
#     model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    # data = load_data("/home/zhangyh/rag_dataset/wikiQA_gpt.json")
    # args = {
    #     "question_lst": ["你是谁？"]*30,
    #     "model_path": "/home/zhangyh/models/Qwen2-7B-Instruct",
    #     "max_bs": 512
    # }
    # result = generate_data(**args)
    # multiprocessing.set_start_method('spawn')
    model_args, data_args, finetuning_args, generating_args = get_infer_args()
    model_args = model_args.to_dict()

    # finetuning_args = finetuning_args.
    print(model_args)
    ans = get_rouge(**model_args)
    print(ans)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


"""
(model_name_or_path='/home/zhangyh/projs/LLaMA-Factory/output_qwen2_0715_e1',
adapter_name_or_path=None,
adapter_folder=None, 
cache_dir=None, 
use_fast_tokenizer=True, 
resize_vocab=False, 
split_special_tokens=False, 
new_special_tokens=None, 
model_revision='main', 
low_cpu_mem_usage=True, 
quantization_method='bitsandbytes', 
quantization_bit=None, 
quantization_type='nf4', 
double_quantization=True, 
quantization_device_map=None, 
rope_scaling=None, 
flash_attn='auto', 
shift_attn=False, 
mixture_of_depths=None, 
use_unsloth=False, 
visual_inputs=False, 
moe_aux_loss_coef=None, 
disable_gradient_checkpointing=False, 
upcast_layernorm=False, 
upcast_lmhead_output=False, 
train_from_scratch=False, 
infer_backend='vllm', 
vllm_maxlen=2048, 
vllm_gpu_util=0.9, 
vllm_enforce_eager=False, 
vllm_max_lora_rank=32, 
offload_folder='offload',
use_cache=True, 
infer_dtype='auto', 
print_param_status=False)
"""
