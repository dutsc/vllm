results_folder="./results"
model="/share/models/Meta-Llama-3-8B-Instruct"
dataset_name="sharegpt"
dataset_path="/share/dataset/ShareGPT52K/sg_90k_part1.json"
num_prompts=3
qps=20

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