python main.py \
  --model_name_or_path /home/zhangyh/models/tora-7b \
  --phase infer \
  --template llama \
  --infer_backend vllm \
  --temperature 0.7 \
  --vllm_maxlen 8192 \
  --vllm_gpu_util 0.9 \
  --vllm_enforce_eager False \
  --data_pth /home/zhangyh/rag_dataset/wikiQA_gpt.json \
  --max_bs 512

  # export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890