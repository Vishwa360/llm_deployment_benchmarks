

export MODEL_ID="Qwen/Qwen3-1.7B"

vllm serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9