import torch 
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizerBase
import json 
import random 
import numpy as np
from vllm.transformers_utils.tokenizer import get_tokenizer

def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(0,
                                         tokenizer.vocab_size,
                                         size=prefix_len).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = [] 
    for i in range(num_prompts):
        prompt = tokenizer.decode(prefix_token_ids +
                                  [(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])])

        input_requests.append((prompt, int(prefix_len + input_lens[i]),
                               int(output_lens[i]), None))

    return input_requests

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
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

    # Shuffle the dataset.
    # random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
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
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset

dataset_path = "/share/dataset/ShareGPT52K/sg_90k_part1.json"
model_path = "/share/models/Meta-Llama-3-8B-Instruct"
tokenizer = get_tokenizer(model_path)
dataset = sample_sharegpt_requests(dataset_path=dataset_path,
                                   num_requests=200,
                                   tokenizer=tokenizer)
# dataset = sample_random_requests(prefix_len=10,
#                                  input_len=16,
#                                  output_len=32,
#                                  num_prompts=200,
#                                  range_ratio=1,
#                                  tokenizer=tokenizer,
#                                 )

input_lens = []
output_lens = []
for idx, req in enumerate(dataset):
    print(f"idx:{idx}, input_len:{req[1]}, output_len:{req[2]}")
    input_lens.append(req[1])
    output_lens.append(req[2])


import numpy as np
import matplotlib.pyplot as plt
input_sorted_data = np.sort(input_lens)
output_sorted_data = np.sort(output_lens)
input_cdf = np.arange(1, len(input_sorted_data) + 1) / len(input_sorted_data)
output_cdf = np.arange(1, len(output_sorted_data) + 1) / len(output_sorted_data)
plt.plot(input_sorted_data, input_cdf, linestyle='-')
plt.plot(output_sorted_data, output_cdf, linestyle='-')
plt.xlabel('Length')
plt.ylabel('Cumulative Probability')
plt.title('CDF of dataset Lengths')
plt.grid(True)
plt.savefig("./sharegpt_cdf.png")

