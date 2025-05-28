results_folder="./results"
model="/share/models/Meta-Llama-3-8B-Instruct"
model="/share/models/Llama-3-8B-Instruct-Gradient-1048k"
# dataset_name="sharegpt"
dataset_name="replicated_data"
dataset_path="/share/dataset/ShareGPT52K/replicated_data.json"
num_prompts=1000
qps=1.5

# for qps in $(seq 1 1 10); do
python3 ../benchmark_serving.py \
        --backend vllm \
        --model $model \
        --dataset-name $dataset_name \
        --dataset-path $dataset_path \
        --num-prompts $num_prompts \
        --port 8006 \
        --save-result \
        --result-dir $results_folder \
        --result-filename "$dataset_name"-qps-"$qps".json \
        --request-rate "$qps"
# done