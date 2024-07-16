python /home/zhangyh/projs/rrag/test/tmp.py \
  --model_name_or_path /home/zhangyh/projs/LLaMA-Factory/output_qwen2_0715_e1 \
  --template qwen \
  --infer_backend vllm \
  --temperature 0.7 \
  --vllm_maxlen=8192 \
  --vllm_gpu_util=0.9 \
  --vllm_enforce_eager=False \
