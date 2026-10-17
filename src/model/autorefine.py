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

from llm_tools.vllm_offline import VLLMOffline
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
    text = text.split('<search>')[0]
    text = text.split('<answer>')[0]
    text.replace("<thiink>", "").replace("</think>", "")
    text = text.strip()
    return text
    
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

class AutoRefine(nn.Module):
    def __init__(self,
                 base_model: VLLMOffline,
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
        
        # Assume that after </answer> is always the eos token, so do not add a answer stopping criteria
        
        if not use_vllm:
            # Initialize the stopping criteria
            self.search_target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n", 
                                            "</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
            self.search_stopping_criteria = StoppingCriteriaList([StopOnSequence(self.search_target_sequences, tokenizer)])
            
            # only for step-wise time measurement
            self.think_target_sequences = ["<search>", "\n<search>", "\n\n<search>", " <search>", "\n <search>", "\n\n <search>", 
                                           "<answer>", "\n<answer>", "\n\n<answer>", " <answer>", "\n <answer>", "\n\n <answer>"]
            self.think_stopping_criteria = StoppingCriteriaList([StopOnSequence(self.think_target_sequences, tokenizer)])
            
            self.answer_target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
        else: 
            self.search_stopping_sequences = ["</search>", "</answer>"]
            
            # only for step-wise time measurement
            self.think_stopping_sequences = ["<search>", "<answer>"]
            
    def reset_prefix_cache(self):
        if self.use_vllm:
            self.base_model.reset_prefix_cache()

    @torch.inference_mode()
    def _text_generation(self, all_prompts, stopping_criteria=None):
        metadata = {'num_request': len(all_prompts)}
        
        input_batch = self.tokenizer(
            all_prompts,
            add_special_tokens=False,
            return_attention_mask=True,
            padding=True,
            truncation=False,
            return_tensors="pt"
        ).to(self.base_model.device)
        metadata['prompt_tokens'] = (input_batch['input_ids'] != self.tokenizer.pad_token_id).sum().item()

        outputs = self.base_model.generate(
            **input_batch,
            max_new_tokens=self.max_gen_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=None, top_p=None, top_k=None,
        )

        outputs = outputs[:, input_batch['input_ids'].shape[1]:]  # Remove the input part
        metadata['completion_tokens'] = (outputs != self.tokenizer.pad_token_id).sum().item()
        all_responses = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        
        return all_responses, metadata

    def _text_generation_vllm(self, all_prompts, stopping_sequences=None):
        all_responses, metadata = self.base_model.batch_infer_with_prompt_texts(
            all_prompts, max_tokens=self.max_gen_tokens, stopping_sequences=stopping_sequences)

        return all_responses, metadata

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
            assert not any(item.chunk_items for item in infer_inputs), "AutoRefine should not have any retrieved chunks in the first turn."
            
            # Initial prompt
            prompt = ("You are a helpful assistant excel at answering questions with multi-turn search engine calling. "
                      "To answer questions, you must first reason through the available information using <think> and </think>. "
                      "If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. "
                      "After each search, you need to summarize and refine the existing documents in <refine> and </refine>. "
                      "You may send multiple search requests if needed. "
                      "Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>. Question: {question}\n")
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
            curr_search_template = '\n\n<documents>{search_results}</documents>\n\n'
            search_texts = [curr_search_template.format(search_results=text) for text in search_texts]
            
            all_prompts = [prompt + search_text for prompt, search_text in zip(all_prompts, search_texts)]

        if self.use_vllm:
            all_responses, metadata = self._text_generation_vllm(all_prompts, self.search_stopping_sequences)
        else: 
            all_responses, metadata = self._text_generation(all_prompts, self.search_stopping_criteria)

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

                if self.use_vllm:
                    unfinished_all_responses, unfinished_metadata = self._text_generation_vllm(list(unfinished_prompts.values()))
                else:
                    unfinished_all_responses, unfinished_metadata = self._text_generation(list(unfinished_prompts.values()))
                
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
        for item in infer_outputs:
            item.infer_time_each_steps.append(total_time)

        return infer_outputs, metadata

    @torch.inference_mode()
    def _text_generation_with_kv_cache(self, prompt, past_key_values=None, max_gen_tokens=64, stopping_target_sequences=[], input_end_token=False):        
        input_token_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(self.base_model.device)
        
        metadata = dict()
        metadata['prompt_tokens'] = input_token_ids.size(1)
        metadata['completion_tokens'] = 0

        outputs = self.base_model(
            input_ids=input_token_ids,
            past_key_values=past_key_values, 
            output_hidden_states=True,
        )
        
        all_token_ids = input_token_ids[0]
        past_key_values = outputs.past_key_values
        input_token_ids = torch.argmax(outputs.logits[0:1, -1:, :], dim=-1,keepdim=False)
        token_id = input_token_ids.view(-1).long()
        all_token_ids = torch.cat([all_token_ids, token_id], dim=0)
        
        if getattr(self, 'target_ids', None) is None:
            self.target_ids = dict()
        stop_target_ids = []
        stop_last_target_ids = []
        for target_sequence in stopping_target_sequences:
            if target_sequence not in self.target_ids:
                self.target_ids[target_sequence] = self.tokenizer.encode(target_sequence, add_special_tokens=False)
            stop_target_ids.append(torch.tensor(self.target_ids[target_sequence], dtype=torch.long, device=self.base_model.device))
            stop_last_target_ids.append(self.target_ids[target_sequence][-1])
        
        stop_flag = False
        if all_token_ids[-1].item() == self.tokenizer.eos_token_id:
            stop_flag = True
        elif all_token_ids[-1].item() in stop_last_target_ids:
            for target_ids in stop_target_ids:
                target_len = target_ids.numel()
                if all_token_ids.numel() >= target_len and torch.equal(all_token_ids[-target_len:], target_ids):
                    stop_flag = True
                    break
        
        while not stop_flag and (all_token_ids.numel() < max_gen_tokens):
            outputs = self.base_model(
                input_ids=input_token_ids,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            input_token_ids = torch.argmax(outputs.logits[0:1, -1:, :], dim=-1,keepdim=False)
            token_id = input_token_ids.view(-1).long()
            all_token_ids = torch.cat([all_token_ids, token_id], dim=0)
            
            if all_token_ids[-1].item() == self.tokenizer.eos_token_id:
                stop_flag = True
            elif all_token_ids[-1].item() in stop_last_target_ids:
                for target_ids in stop_target_ids:
                    target_len = target_ids.numel()
                    if all_token_ids.numel() >= target_len and torch.equal(all_token_ids[-target_len:], target_ids):
                        stop_flag = True
                        break

        output_token_ids = all_token_ids[metadata['prompt_tokens']:]
        num_gen_tokens = output_token_ids.numel()
        output_text = self.tokenizer.decode(output_token_ids.tolist(), skip_special_tokens=True)
            
        if input_end_token: 
            outputs = self.base_model(
                input_ids=torch.tensor([[output_token_ids[-1].item()]], device=self.base_model.device),
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            num_gen_tokens += 1
            
        metadata['completion_tokens'] = num_gen_tokens
        
        end_token = self.tokenizer.decode([output_token_ids[-1].item()], skip_special_tokens=False)
        
        return output_text, past_key_values, metadata, end_token
    
    @torch.inference_mode()
    def _forward_with_kv_cache(self, prompt, past_key_values):
            
        metadata = {'completion_tokens': 1}
        
        input_batch = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=True,
            padding=True,
            truncation=False,
            return_tensors="pt"
        ).to(self.base_model.device)
        metadata['prompt_tokens'] = (input_batch['input_ids'] != self.tokenizer.pad_token_id).sum().item()
            
        outputs = self.base_model(
            **input_batch,
            past_key_values=past_key_values,
        )

        past_key_values = outputs.past_key_values
        end_token_id = torch.argmax(outputs.logits[0:1, -1:, :], dim=-1,keepdim=False)
        end_token = self.tokenizer.decode([end_token_id.item()], skip_special_tokens=False)

        return past_key_values, metadata, end_token
    
    # inference
    def generate_with_step_wise_time(self, infer_inputs: List[IterationRecord]) -> List[IterationRecord]:
        assert len(infer_inputs) == 1, "Step-wise time measurement only supports batch size 1."
        assert not self.use_vllm, "Step-wise time measurement does not support vLLM."
        
        infer_inputs = infer_inputs[0]
        
        start_time = time.perf_counter()
        
        cur_turn = len(infer_inputs.chunk_items)
        logger.info(f"Current turn: {cur_turn}")
        infer_outputs = infer_inputs
        total_metadata = {'prompt_tokens': 0, 'completion_tokens': 0}
        
        past_key_values = None
        if 'past_key_values' in infer_inputs.states:
            past_key_values = infer_inputs.states['past_key_values']
            
        if past_key_values is None:
            # not initial retrieval
            assert not infer_inputs.chunk_items, "AutoRefine should not have any retrieved chunks in the first turn."
            
            prompt = ("You are a helpful assistant excel at answering questions with multi-turn search engine calling. "
                      "To answer questions, you must first reason through the available information using <think> and </think>. "
                      "If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. "
                      "After each search, you need to summarize and refine the existing documents in <refine> and </refine>. "
                      "You may send multiple search requests if needed. "
                      "Once you have sufficient information, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>. Question: {question}\n")
            prompt = prompt.format(question=infer_inputs.query)
            if self.use_chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], 
                    add_generation_prompt=True, tokenize=False)
        else: 
            search_text = process_chunk_texts([chunk_item.text for chunk_item in infer_inputs.chunk_items[-1]], 
                                              title_in_first_line=self.title_in_first_line)
            curr_search_template = '\n\n<documents>{search_results}</documents>\n\n'
            prompt = curr_search_template.format(search_results=search_text)
            
        prefill_start_time = time.perf_counter()
        past_key_values, prefill_metadata, end_token = self._forward_with_kv_cache(prompt, past_key_values)
        total_metadata['prompt_tokens'] += prefill_metadata['prompt_tokens']
        total_metadata['completion_tokens'] += prefill_metadata['completion_tokens']
        prefill_end_time = time.perf_counter()
        prefill_total_time = prefill_end_time - prefill_start_time
            
        thought_generation_start_time = time.perf_counter()
        prompt = end_token
        think_response, past_key_values, think_metadata, end_token = self._text_generation_with_kv_cache(
            prompt=prompt, 
            past_key_values=past_key_values, 
            max_gen_tokens=self.max_gen_tokens,
            stopping_target_sequences=self.think_target_sequences)
        total_metadata['prompt_tokens'] += think_metadata['prompt_tokens']
        total_metadata['completion_tokens'] += think_metadata['completion_tokens']
        think_response = prompt + think_response
        thought_generation_end_time = time.perf_counter()
        thought_generation_total_time = thought_generation_end_time - thought_generation_start_time
        
        subquery_generation_start_time = time.perf_counter()
        prompt = end_token
        search_response, past_key_values, search_metadata, end_token = self._text_generation_with_kv_cache(
            prompt=prompt, 
            past_key_values=past_key_values, 
            max_gen_tokens=self.max_gen_tokens,
            stopping_target_sequences=self.search_target_sequences, 
            input_end_token=True)
        total_metadata['prompt_tokens'] += search_metadata['prompt_tokens']
        total_metadata['completion_tokens'] += search_metadata['completion_tokens']
        subquery_generation_end_time = time.perf_counter()
        subquery_generation_total_time = subquery_generation_end_time - subquery_generation_start_time
        answer_generation_total_time = None
        
        response = think_response + search_response
        
        new_thought = get_thoughts(response)
        answer = get_answer(response)
        # if an answer is provided, then no additional subqueries
        new_subquery = get_subquery(response) if answer is None else None
        # if no additional subquery is provided and no answer, then set the answer to "" (fail)
        answer = "" if answer is None and new_subquery is None else answer
        force_answer_flag = False
        
        if answer is not None:
            answer_generation_total_time = subquery_generation_total_time
            subquery_generation_total_time = None
        
        force_answer = cur_turn >= self.max_turns
        if force_answer and answer is None:
            answer_generation_start_time = time.perf_counter()
            
            prompt = "<answer>"
            unfinished_response, _, unfinished_metadata, _ = self._text_generation_with_kv_cache(
                prompt=prompt, 
                past_key_values=past_key_values,
                max_gen_tokens=self.max_gen_tokens,
                stopping_target_sequences=self.answer_target_sequences,
            )
            answer_generation_end_time = time.perf_counter()
            answer_generation_total_time = answer_generation_end_time - answer_generation_start_time
            
            unfinished_response = "<answer>" + unfinished_response
            unfinished_response = unfinished_response.split("</answer>")[0] + "</answer>" if "</answer>" in unfinished_response else unfinished_response
            unfinished_answer = get_answer(unfinished_response)
            # force finish
            unfinished_answer = unfinished_answer if unfinished_answer is not None else ""
            
            answer = unfinished_answer
            new_subquery = None
            force_answer_flag = True

            total_metadata['prompt_tokens'] += unfinished_metadata['prompt_tokens']
            total_metadata['completion_tokens'] += unfinished_metadata['completion_tokens']

        infer_outputs = deepcopy(infer_inputs)
        infer_outputs.thoughts.append(new_thought)
        infer_outputs.subqueries.append(new_subquery)
        infer_outputs.answer = answer
        infer_outputs.force_answer = force_answer_flag

        if infer_outputs.answer is None:
            infer_outputs.states = {
                "past_key_values": past_key_values
            }
        else:
            infer_outputs.states = {}

        end_time = time.perf_counter()
        total_time = end_time - start_time
        infer_outputs.infer_time_each_steps.append(total_time)
        
        if 'step_wise_time' not in infer_outputs.extra_info:
            infer_outputs.extra_info['step_wise_time'] = dict()
        if 'prefill_time' not in infer_outputs.extra_info['step_wise_time']:
            infer_outputs.extra_info['step_wise_time']['prefill_time'] = []
        if 'thought_generation_time' not in infer_outputs.extra_info['step_wise_time']:
            infer_outputs.extra_info['step_wise_time']['thought_generation_time'] = []
        if 'subquery_generation_time' not in infer_outputs.extra_info['step_wise_time']:
            infer_outputs.extra_info['step_wise_time']['subquery_generation_time'] = []
        if 'answer_generation_time' not in infer_outputs.extra_info['step_wise_time']:
            infer_outputs.extra_info['step_wise_time']['answer_generation_time'] = []
        infer_outputs.extra_info['step_wise_time']['prefill_time'].append(prefill_total_time)
        infer_outputs.extra_info['step_wise_time']['thought_generation_time'].append(thought_generation_total_time)
        infer_outputs.extra_info['step_wise_time']['subquery_generation_time'].append(subquery_generation_total_time)
        infer_outputs.extra_info['step_wise_time']['answer_generation_time'].append(answer_generation_total_time)

        return [infer_outputs], total_metadata

    @classmethod
    def load(cls, config):
        torch_dtype = torch.float32
        if getattr(config.training, 'bf16', False): 
            torch_dtype = torch.bfloat16
        elif getattr(config.training, 'fp16', False):
            torch_dtype = torch.float16
        
        tokenizer = load_tokenizer(tokenizer_name_or_path=config.model.model_name_or_path)
        if config.vllm.use_vllm:
            base_model = VLLMOffline(config.model.model_name_or_path, seed=config.training.seed, 
                                     gpu_memory_utilization=config.vllm.qa_model_gpu_memory_utilization,
                                     max_model_len=config.vllm.qa_model_max_model_len,
                                     max_num_seqs=config.vllm.qa_model_max_num_seqs,
                                     dtype=torch_dtype)
        else:
            base_model = AutoModelForCausalLMSafeLoader.from_pretrained(config.model.model_name_or_path, 
                                                                        torch_dtype=torch_dtype, 
                                                                        device_map="auto",
                                                                        attn_implementation=config.model.attn_implementation)
        
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
