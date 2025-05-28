
MODEL_PATH="/share/models/Meta-Llama-3-8B-Instruct"

CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH  \
    --port 8102 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8