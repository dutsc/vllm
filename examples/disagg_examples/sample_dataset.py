import json 
import numpy as np 
from transformers import PreTrainedTokenizerBase
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple
from vllm.transformers_utils.tokenizer import get_tokenizer

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int = None,
    tokenizer: PreTrainedTokenizerBase = None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # print(len(dataset))
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    if num_requests is None:
        num_requests = len(dataset)
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        # if prompt_len < 2047 or prompt_len > 16384:
        #     continue 
        # if output_len > 256:
        #     continue
        
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset

def draw_cdf(data):
    import matplotlib.pyplot as plt
    def extract_lengths(data):
        input_lengths = [item[1] for item in data]
        output_lengths = [item[2] for item in data]
        return input_lengths, output_lengths

    # 计算 CDF
    def compute_cdf(lengths):
        hist, bin_edges = np.histogram(lengths, bins=100, density=True)
        cdf = np.cumsum(hist * np.diff(bin_edges))
        return bin_edges[1:], cdf

    # 绘制 CDF 图
    def plot_cdf(input_edges, input_cdf, output_edges, output_cdf):
        plt.figure(figsize=(10, 6))
        plt.plot(input_edges, input_cdf, label='Input Length CDF', color='blue')
        plt.plot(output_edges, output_cdf, label='Output Length CDF', color='orange')
        plt.xlabel('Length')
        plt.ylabel('CDF')
        plt.title('CDF of Input and Output Lengths')
        plt.legend()
        plt.grid(True)
        plt.savefig("./cdf_sg_90k_part1.png")
        
    input_lengths, output_lengths = extract_lengths(data)
    
    input_edges, input_cdf = compute_cdf(input_lengths)
    output_edges, output_cdf = compute_cdf(output_lengths)
    
    plot_cdf(input_edges, input_cdf, output_edges, output_cdf)

dataset="/share/dataset/ShareGPT52K/sg_90k_part1.json"
model="/share/models/Meta-Llama-3-8B-Instruct"
tokenizer = get_tokenizer(model)
input_requests = sample_sharegpt_requests(
    dataset_path=dataset,
    tokenizer=tokenizer,
)

draw_cdf(data=input_requests)


# num_prompts = 1000

# input_requests = sample_sharegpt_requests(
#     dataset_path=dataset,
#     num_requests=num_prompts,
#     tokenizer=tokenizer,
# )

# print(len(input_requests))

# data = [
#     {"prompt": prompt, "input_len": prompt_len, "output_len": output_len}
#     for (prompt, prompt_len, output_len, extra) in input_requests
# ]

# replicated_data = data * 100
# output_file = "replicated_data.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(replicated_data, f, indent=4)