import os
import re
import logging
import time
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Optional, List
from transformers import PreTrainedTokenizer
from transformers import StoppingCriteriaList, StoppingCriteria
import asyncio

from llm_tools.vllm_offline_async import VLLMOfflineAsync
from model.infer_inputs_outputs import IterationRecord
from llm_tools.safe_loader import AutoModelForCausalLMSafeLoader, AutoTokenizerSafeLoader

logger = logging.getLogger(__name__)


# Define the custom stopping criterion
class StopOnSequence(StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        targets = [torch.tensor(t, device=device) for t in self.target_ids]

        stop_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for b in range(batch_size):
            seq = input_ids[b]
            for target, target_len in zip(targets, self.target_lengths):
                if seq.shape[0] >= target_len and torch.equal(seq[-target_len:], target):
                    stop_flags[b] = True
                    break
        return stop_flags
    
def get_subquery(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None
    
def get_thoughts(text):
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = pattern.findall(text)
    matches = [item.strip() for item in matches if item.strip() != '']
    if matches and len(matches) > 0:
        # save all thoughts (sometimes the model generates multiple <think></think> thoughts)
        return ' '.join(matches)
    else:
        return None
    
def get_answer(text):
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None
    
def process_chunk_texts(texts, title_in_first_line=True): 
    format_reference = ''
    for idx, text in enumerate(texts):
        if title_in_first_line:
            title = text.split("\n")[0]
            text = "\n".join(text.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        else:
            format_reference += f"[Doc {idx+1}]\n{text}\n"

    return format_reference


def load_tokenizer(tokenizer_name_or_path):
    tokenizer = AutoTokenizerSafeLoader.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token is None:
        tokenizer.bos_token = ''
    tokenizer.padding_side = 'left'

    return tokenizer

class AsyncVLLMSearchR1(nn.Module):
    def __init__(self,
                 base_model,
                 tokenizer: PreTrainedTokenizer,
                 max_turns: int,
                 use_chat_template: bool,
                 max_gen_tokens: int,
                 use_vllm: bool, 
                 config):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.use_chat_template = use_chat_template
        self.max_gen_tokens = max_gen_tokens
        self.use_vllm = use_vllm
        self.config = config
        
        self.title_in_first_line = getattr(self.config.data, 'title_in_first_line', True)
        
        self.search_target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n", 
                                        "</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
        self.think_target_sequences = ["<search>", "\n<search>", "\n\n<search>", " <search>", "\n <search>", "\n\n <search>", 
                                      "<answer>", "\n<answer>", "\n\n<answer>", " <answer>", "\n <answer>", "\n\n <answer>"]
        self.answer_target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
        
        # Assume that after </answer> is always the eos token, so do not add a answer stopping criteria
        self.search_stopping_sequences = ["</search>"]
        
        # only for step-wise time measurement
        self.think_stopping_sequences = ["</think>"]
            
    def reset_prefix_cache(self):
        self.base_model.reset_prefix_cache()

    def _text_generation_vllm(self, all_prompts, stopping_sequences=None):
        metadata = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "num_request": 0, 
            "max_length": 0
        }

        all_responses = [None for _ in range(len(all_prompts))]
        all_latencies = [0 for _ in range(len(all_prompts))]

        results =  self.base_model.batch_infer_with_prompt_texts(
            all_prompts, max_tokens=self.max_gen_tokens, stopping_sequences=stopping_sequences
        ) 
        for result in results:
            metadata["prompt_tokens"] += result["prompt_tokens"]
            metadata["completion_tokens"] += result["completion_tokens"]
            metadata["num_request"] += 1
            metadata["max_length"] = max(metadata["max_length"], result["prompt_tokens"] + result["completion_tokens"])

            all_responses[result["index"]] = result["response"]
            all_latencies[result["index"]] = result["latency"]

        return all_responses, metadata, all_latencies
    

    def batch_generate(self, infer_inputs: List[IterationRecord]) -> List[IterationRecord]:
        
        start_time = time.perf_counter()
        
        batch_size = len(infer_inputs)
        cur_turn = len(infer_inputs[0].chunk_items)
        logger.info(f"Current turn: {cur_turn}, batch size: {batch_size}")

        past_prompts = None
        if 'past_prompt' in infer_inputs[0].states:
            past_prompts = [item.states['past_prompt'] for item in infer_inputs]

        if past_prompts is None:
            # not initial retrieval
            assert not any(item.chunk_items for item in infer_inputs), "Search-R1 should not have any retrieved chunks in the first turn."
            
            # Initial prompt
            prompt = ("Answer the given question. "
                    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
                    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. "
                    "You can search as many times as your want. "
                    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n")
            all_prompts = [prompt.format(question=item.query) for item in infer_inputs]
            if self.use_chat_template:
                all_prompts = self.tokenizer.apply_chat_template(
                    [[{"role": "user", "content": p}] for p in all_prompts], 
                    add_generation_prompt=True, tokenize=False)
        else:
            all_prompts = past_prompts
            search_texts = [
                process_chunk_texts([chunk_item.text for chunk_item in item.chunk_items[-1]],
                                    title_in_first_line=self.title_in_first_line) for item in infer_inputs
            ]
            curr_search_template = '\n\n<information>{search_results}</information>\n\n'
            search_texts = [curr_search_template.format(search_results=text) for text in search_texts]
            
            all_prompts = [prompt + search_text for prompt, search_text in zip(all_prompts, search_texts)]

        all_responses, metadata, all_latencies = self._text_generation_vllm(all_prompts, self.search_stopping_sequences)

        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            all_record_responses = [[out] for out in all_responses]

        # remove additional tokens
        all_responses = [out.split("</search>")[0] + "</search>" if "</search>" in out else out
                           for out in all_responses]
        all_responses = [out.split("</answer>")[0] + "</answer>" if "</answer>" in out else out
                           for out in all_responses]

        past_prompts = [prompt + output for prompt, output in zip(all_prompts, all_responses)]
        all_new_thoughts = [get_thoughts(output) for output in all_responses]
        all_answers = [get_answer(output) for output in all_responses]
        # if an answer is provided, then no additional subqueries
        all_new_subqueries = [get_subquery(output) if all_answers[i] is None else None for i, output in enumerate(all_responses)]
        # if no additional subquery is provided and no answer, then set the answer to "" (fail)
        all_answers = ["" if ans is None and all_new_subqueries[i] is None else ans for i, ans in enumerate(all_answers)]
        all_force_answer_flags = [False for _ in range(batch_size)]
        
        force_answer = cur_turn >= self.max_turns
        if force_answer:
            unfinished_prompts = {i: past_prompts[i] for i, ans in enumerate(all_answers) if ans is None or ans == ""}
            if len(unfinished_prompts) > 0:
                for i in unfinished_prompts:
                    if unfinished_prompts[i].endswith("</search>"):
                        unfinished_prompts[i] = unfinished_prompts[i].rsplit("<search>", 1)[0]
                    # force answer
                    unfinished_prompts[i] += "<answer>"

                unfinished_all_responses, unfinished_metadata, unfinished_all_latencies = self._text_generation_vllm(list(unfinished_prompts.values()), self.search_stopping_sequences)

                if os.getenv("DEBUG_MODE", "false").lower() == "true":
                    unfinished_all_record_responses = [[out] for out in unfinished_all_responses]
                
                unfinished_all_responses = ["<answer>" + out for out in unfinished_all_responses]
                unfinished_all_responses = [out.split("</answer>")[0] + "</answer>" if "</answer>" in out else out
                                            for out in unfinished_all_responses]
                unfinished_all_answers = [get_answer(output) for output in unfinished_all_responses]
                # force finish
                unfinished_all_answers = [ans if ans is not None else "" for ans in unfinished_all_answers]

                for cur_idx, global_idx in enumerate(unfinished_prompts.keys()):
                    all_answers[global_idx] = unfinished_all_answers[cur_idx]
                    all_new_subqueries[global_idx] = None
                    past_prompts[global_idx] = unfinished_prompts[global_idx].removesuffix("<answer>") + unfinished_all_responses[cur_idx]
                    all_force_answer_flags[global_idx] = True
                    all_latencies[global_idx] = unfinished_all_latencies[cur_idx]
                    if os.getenv("DEBUG_MODE", "false").lower() == "true":
                        all_record_responses[global_idx].extend(unfinished_all_record_responses[cur_idx])

                metadata['num_request'] += unfinished_metadata['num_request']
                metadata['prompt_tokens'] += unfinished_metadata['prompt_tokens']
                metadata['completion_tokens'] += unfinished_metadata['completion_tokens']

        infer_outputs = deepcopy(infer_inputs)
        for i, item in enumerate(infer_outputs):
            item.thoughts.append(all_new_thoughts[i])
            item.subqueries.append(all_new_subqueries[i])
            item.answer = all_answers[i]
            item.force_answer = all_force_answer_flags[i]
            
            if os.getenv("DEBUG_MODE", "false").lower() == "true":
                record_responses = item.states.get("record_responses", [])
                record_responses.extend(all_record_responses[i])
            
            if item.answer is None:
                item.states = {"past_prompt": past_prompts[i]}
            else:
                item.states = {}
                
            if os.getenv("DEBUG_MODE", "false").lower() == "true":
                item.states["record_responses"] = record_responses

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / batch_size
        for i, item in enumerate(infer_outputs):
            item.infer_time_each_steps.append(all_latencies[i])

        return infer_outputs, metadata
    
    @classmethod
    def load(cls, config):
        torch_dtype = torch.float32
        if getattr(config.training, 'bf16', False): 
            torch_dtype = torch.bfloat16
        elif getattr(config.training, 'fp16', False):
            torch_dtype = torch.float16
        
        tokenizer = load_tokenizer(tokenizer_name_or_path=config.model.model_name_or_path)

        base_model = VLLMOfflineAsync(config.model.model_name_or_path, seed=config.training.seed, 
                                    gpu_memory_utilization=config.vllm.qa_model_gpu_memory_utilization,
                                    max_model_len=config.vllm.qa_model_max_model_len,
                                    max_num_seqs=config.vllm.qa_model_max_num_seqs,
                                    dtype=torch_dtype)

        model = cls(
            base_model=base_model,
            tokenizer=tokenizer,
            max_turns=config.model.max_turns,
            use_chat_template=config.model.use_chat_template,
            max_gen_tokens=config.model.max_gen_tokens, 
            use_vllm=config.vllm.use_vllm,
            config=config
        )

        return model
