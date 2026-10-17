import json
from hashlib import md5
import numpy as np
import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Callable, Optional, Union
import os
import torch
from safetensors.torch import load_file

from torch import nn, Tensor

import logging
logger = logging.getLogger(__name__)

def log_example_data(input_data, tokenizer, label_pad_token_id):
    if input_data is None:
        logger.info("Input data is None, skipping logging example data.")
        return
    
    logger.info("############ Example data:\n")
    example_token_ids = input_data['input_ids'][0].tolist()
    while example_token_ids and example_token_ids[0] == tokenizer.pad_token_id:
        example_token_ids.pop(0)
    while example_token_ids and example_token_ids[-1] == tokenizer.pad_token_id:
        example_token_ids.pop()
    example_tokens = tokenizer.decode(example_token_ids)
    logger.info(example_tokens)
    if 'labels' in input_data:
        logger.info("############ Example labels:\n")
        example_labels = input_data['labels'][0].tolist()
        current_line = []
        for token_id in example_labels:
            if token_id == label_pad_token_id:
                if current_line:
                    logger.info(tokenizer.decode(current_line))
                    current_line = []
            else:
                current_line.append(token_id)
        if current_line:  # the last line
            logger.info(tokenizer.decode(current_line))

def list_unflatten(flat_list: List, length: List) -> List[List]:
    result = []
    index = 0
    for l in length:
        result.append(flat_list[index: index + l])
        index += l
    
    return result

def list_find(lst, value, start=0):
    for i, x in enumerate(lst[start:], start):
        if x == value:
            return i
    return -1

def json_compact(json_str: str) -> str:
    """
    Compact a JSON string by removing unnecessary whitespace.
    """
    return json.dumps(json.loads(json_str), separators=(',', ':'))


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()

def compute_embedding_mdhash_id(embedding: Union[List[float], np.ndarray], prefix: str = "") -> str:
    if isinstance(embedding, np.ndarray):
        embedding = embedding.flatten().tolist()

    rounded_embedding = [round(x, 6) for x in embedding]
    json_str = json.dumps(rounded_embedding, separators=(",", ":"), ensure_ascii=False)
    return prefix + md5(json_str.encode()).hexdigest()

def normalize_answer(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


# Metrics for retrieval evaluation


# Metrics for qa evaluation
# MRQA official eval
def calculate_em_score(gold_list: List[str], predicted: str, aggregation_fn: Callable = np.max) -> float:
    normalized_predicted = normalize_answer(predicted)
    em_scores = [1.0 if normalize_answer(gold) == normalized_predicted else 0.0 for gold in gold_list]
    return aggregation_fn(em_scores)

def calculate_cover_em_score(gold_list: List[str], predicted: str, aggregation_fn: Callable = np.max) -> float:
    normalized_predicted = normalize_answer(predicted)
    cover_em_scores = [1.0 if normalize_answer(gold) in normalized_predicted else 0.0 for gold in gold_list]
    return aggregation_fn(cover_em_scores)

def compute_f1(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)

def calculate_f1_score(gold_list: List[str], predicted: str, aggregation_fn: Callable = np.max) -> float:
    f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
    return aggregation_fn(f1_scores)

def calculate_search_success_score(golden_answers: List[str], retrieved_passages: List[str]) -> bool:
    normalized_golden_answers = [normalize_answer(ans) for ans in golden_answers]
    normalized_retrieved_passages = [normalize_answer(passage) for passage in retrieved_passages]
    
    for ans in normalized_golden_answers:
        for passage in normalized_retrieved_passages:
            if ans in passage:
                return True
    return False


def load_model_weights(model, path, name) -> bool: 
    if path is None:
        return False
    
    # try safetensors first
    model_weights_path = os.path.join(path, f"{name}.safetensors")
    if os.path.exists(model_weights_path):
        state_dict = load_file(model_weights_path)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_weights_path}")
        return True

    # if not exists, try bin
    model_path = os.path.join(path, f"{name}.bin")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_path}")
        return True
    
    return False

# def maybe_zero_3(param, ignore_status=False, name=None):
#     from deepspeed import zero
#     from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
#     if hasattr(param, "ds_id"):
#         if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
#             if not ignore_status:
#                 print(name, 'no ignore status')
#         with zero.GatheredParameters([param]):
#             param = param.data.detach().cpu().clone()
#     else:
#         param = param.detach().cpu().clone()
#     return param

# def get_adapter_state_maybe_zero_3(named_params, keys_to_match):
#     to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
#     to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
#     return to_return

class DotDict(dict):
    """Dict that supports dot notation as well as dictionary access notation."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")


# model utils
def compute_segment_lm_metrics(logits, labels, input_ids, 
                               start_token_id=None, end_token_id=None, spec_token_ids=None,
                               ignore_index=-100):
    B, T = input_ids.shape
    device = input_ids.device
    
    range_tensor = torch.arange(T, device=device).unsqueeze(0)
    segment_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    if start_token_id is not None:
        start_pos = (input_ids == start_token_id).float().argmax(dim=1)
        segment_mask &= range_tensor >= start_pos.unsqueeze(1)
        
    if end_token_id is not None:
        end_pos = (input_ids == end_token_id).float().argmax(dim=1)
        segment_mask &= range_tensor <= end_pos.unsqueeze(1)

    masked_labels = labels.clone()
    masked_labels[~segment_mask] = ignore_index
    
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = masked_labels[..., 1:].contiguous().view(-1)

    valid_mask = shift_labels != ignore_index
    shift_logits = shift_logits[valid_mask]
    shift_labels = shift_labels[valid_mask]
    
    loss = nn.functional.cross_entropy(shift_logits, shift_labels)
    
    preds = shift_logits.argmax(dim=-1)
    accuracy = (preds == shift_labels).float().mean()

    accuracy_without_spec_tokens = None
    if spec_token_ids is not None:
        spec_mask = ~torch.isin(shift_labels, spec_token_ids)
        accuracy_without_spec_tokens = (
            ((preds == shift_labels) & spec_mask).float().sum()
            / spec_mask.float().sum()
        )
        
    return {
        "loss": loss,
        "accuracy": accuracy,
        "accuracy_without_spec_tokens": accuracy_without_spec_tokens,
    }

def count_parameters(model):
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e9:
        return f"{n / 1e9:.4f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.4f}M"
    else:
        return f"{n:,}"
    
def flatten_list(x):
    if isinstance(x, list):
        res = []
        for item in x:
            res.extend(flatten_list(item))
        return res
    else:
        return [x]
    
def unflatten_list(flat_list, sizes):
    if sizes is None:
        return flat_list
    out, idx = [], 0
    for s in sizes:
        out.append(flat_list[idx: idx + s])
        idx += s
    return out
