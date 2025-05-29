#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.
# MODEL_PATH="/share/models/Meta-Llama-3-8B-Instruct"
MODEL_PATH="/share/models/Llama-3-8B-Instruct-Gradient-1048k"
# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH \
    --port 8101 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":6,"producer_num":2,"consumer_num":4}' &

CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_PATH \
    --port 8102 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_producer","kv_rank":1,"kv_parallel_size":6,"producer_num":2,"consumer_num":4}' &

# CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_PATH \
#     --port 8103 \
#     --max-model-len 16384 \
#     --gpu-memory-utilization 0.8 \
#     --kv-transfer-config \
#     '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_producer","kv_rank":2,"kv_parallel_size":6,"producer_num":4,"consumer_num":2}' &

# CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH \
#     --port 8104 \
#     --max-model-len 16384 \
#     --gpu-memory-utilization 0.8 \
#     --kv-transfer-config \
#     '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_producer","kv_rank":3,"kv_parallel_size":6,"producer_num":4,"consumer_num":2}' &

# CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_PATH \
#     --port 8105 \
#     --max-model-len 16384 \
#     --gpu-memory-utilization 0.8 \
#     --kv-transfer-config \
#     '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_producer","kv_rank":4,"kv_parallel_size":6,"producer_num":5,"consumer_num":1}' &


# decoding instance, which is the KV consumer

CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_PATH \
    --port 8201 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_consumer","kv_rank":2,"kv_parallel_size":6,"producer_num":2,"consumer_num":4}' &

CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH \
    --port 8202 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_consumer","kv_rank":3,"kv_parallel_size":6,"producer_num":2,"consumer_num":4}' &


CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_PATH \
    --port 8203 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_consumer","kv_rank":4,"kv_parallel_size":6,"producer_num":2,"consumer_num":4}' &

CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_PATH \
    --port 8204 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"XpYdNcclConnector","kv_role":"kv_consumer","kv_rank":5,"kv_parallel_size":6,"producer_num":2,"consumer_num":4}' &

# wait until prefill and decode instances are ready
wait_for_server 8101
wait_for_server 8102
# wait_for_server 8103
# wait_for_server 8104
# wait_for_server 8105
wait_for_server 8201
wait_for_server 8202
wait_for_server 8203
wait_for_server 8204

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens 
#   to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM 
#   instance
# NOTE: the usage of this API is subject to change --- in the future we will 
# introduce "vllm connect" to connect between prefill and decode instances
python3 ../../benchmarks/disagg_benchmarks/xpyd_disagg_prefill_proxy_server.py &
python3 ../../benchmarks/disagg_benchmarks/zmq_proxy.py &
sleep 1


