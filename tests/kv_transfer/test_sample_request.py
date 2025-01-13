import torch 
from typing import Any, List, Optional, Tuple
from transformers import PreTrainedTokenizerBase 
import json 
import numpy as np 
from vllm.transformers_utils.tokenizer import get_tokenizer

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


dataset_path="/share/dataset/ShareGPT52K/sg_90k_part1.json"
model_path = "/share/models/Meta-Llama-3-8B-Instruct"
tokenizer = get_tokenizer(model_path) 
dataset = sample_sharegpt_requests(dataset_path=dataset_path,
                                   num_requests=5,
                                   tokenizer=tokenizer)

input_lens = []
output_lens = []
for idx, req in enumerate(dataset):
    print(f"idx:{idx}, input_len:{req[1]}, output_len:{req[2]}")
    input_lens.append(req[1])
    output_lens.append(req[2])