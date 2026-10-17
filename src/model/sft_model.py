import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple, Any
import numpy as np
import time
import random
from omegaconf import DictConfig
from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import ModelOutput
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from safetensors.torch import save_file

from llm_tools.vllm_offline import VLLMOffline
from model.infer_inputs_outputs import IterationRecord
from llm_tools.safe_loader import (
    AutoModelForCausalLMSafeLoader,
    AutoModelSafeLoader, 
    AutoTokenizerSafeLoader, 
    AutoConfigSafeLoader, 
    LoraConfigSafeLoader
)
from utils import load_model_weights, count_parameters

import logging
logger = logging.getLogger(__name__)


def load_tokenizer_and_add_special_tokens(config): 
    tokenizer = AutoTokenizerSafeLoader.from_pretrained(
        config.model.model_name_or_path,
        use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token is None:
        tokenizer.bos_token = ''
    tokenizer.padding_side = 'left'

    _, spec_token_list = SFTModel.get_special_added_tokens(config)
    tokenizer.add_special_tokens({"additional_special_tokens": spec_token_list})

    return tokenizer

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

@dataclass
class TrainingOutput(ModelOutput):
    loss: Optional[Tensor] = None
    gen_loss: Optional[Tensor] = None
    self_decode_thought_gen_loss: Optional[Tensor] = None
    self_decode_subquery_gen_loss: Optional[Tensor] = None
    align_loss: Optional[Tensor] = None
    align_mse_loss: Optional[Tensor] = None
    align_unnorm_mse_loss: Optional[Tensor] = None
    align_top1_nce_loss: Optional[Tensor] = None
    align_topk_nce_loss: Optional[Tensor] = None
    align_max_pos_nce_loss: Optional[Tensor] = None
    align_min_pos_nce_loss: Optional[Tensor] = None
    align_per_topk_nce_loss: Optional[Tensor] = None
    align_kl_rank_loss: Optional[Tensor] = None
    self_decode_thought_gen_accuracy: Optional[Tensor] = None
    self_decode_thought_gen_accuracy_without_spec_tokens: Optional[Tensor] = None
    self_decode_subquery_gen_accuracy: Optional[Tensor] = None
    self_decode_subquery_gen_accuracy_without_spec_tokens: Optional[Tensor] = None
    align_top1_accuracy: Optional[Tensor] = None
    align_topk_accuracy: Optional[Tensor] = None
    align_max_pos_accuracy: Optional[Tensor] = None
    align_cosine_similarity: Optional[Tensor] = None
    align_top1_cosine_similarity: Optional[Tensor] = None
    align_topk_cosine_similarity: Optional[Tensor] = None
    align_max_pos_cosine_similarity: Optional[Tensor] = None
    gen_accuracy: Optional[Tensor] = None
    gen_accuracy_without_spec_tokens: Optional[Tensor] = None

class SFTModel(nn.Module):
    def __init__(self, 
                 base_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase, 
                 config: DictConfig, 
                 ):
        super().__init__()
        self.conf = config  # conf is the config of SFT
        self.use_vllm = config.vllm.use_vllm
        self.base_model = base_model
        
        self.title_in_first_line = getattr(self.conf.data, 'title_in_first_line', True)
        
        self.spec_tokens, _ = self.get_special_added_tokens(config)
        self.spec_token_ids = {name: tokenizer.convert_tokens_to_ids(token) 
                               for name, token in self.spec_tokens.items()}
        self.spec_token_id_tensor = torch.tensor(sorted({
            t for v in self.spec_token_ids.values() for t in (v if isinstance(v, list) else [v])
        }), device='cuda')
        
        self.spec_prompts = self.get_special_prompts(config)

        self.max_turns = config.model.max_turns
        self.max_gen_tokens = config.model.max_gen_tokens
        self.retriever_use_fp16 = config.retriever.use_fp16

        self.tokenizer = tokenizer

        self.use_chat_template = config.model.use_chat_template

        self.update_ddp_state()
    
    def update_ddp_state(self):
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
    
    def set_reference_encoder(self, reference_encoder): 
        self.reference_encoder = reference_encoder
        
    def get_reference_encoder(self): 
        return self.reference_encoder
            
    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, 
                llm_input_batches: Dict[str, Any] = None, 
                retrieval_data: Dict[str, Any] = None) -> TrainingOutput:
        input_batch = llm_input_batches['input_batch']
        
        # forward LLM
        outputs = self.base_model(
            input_ids=input_batch['input_ids'],
            attention_mask=input_batch['attention_mask'],
            position_ids=input_batch['position_ids'],
            output_hidden_states=False,
            use_cache=False
        )
        logits = outputs.logits

        # calculate gen_loss
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = input_batch['labels'][..., 1:].contiguous().view(-1)
        # mask
        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]
        gen_num_elements = torch.tensor(shift_logits.shape[0], device=shift_logits.device, dtype=torch.long)
        gen_loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, 
            label_smoothing=self.conf.custom_training.get('label_smoothing', 0.0)
        )
        with torch.no_grad():
            # calculate gen accuracy
            shift_predictions = shift_logits.argmax(dim=-1)
            gen_accuracy = (shift_predictions == shift_labels).float().mean()
            # calculate gen accuracy without special tokens
            spec_token_mask = ~torch.isin(shift_labels, self.spec_token_id_tensor)
            gen_num_elements_without_spec_tokens = spec_token_mask.float().sum().to(dtype=torch.long)
            gen_accuracy_without_spec_tokens = ((shift_predictions == shift_labels) & spec_token_mask).float().sum() / gen_num_elements_without_spec_tokens

        del outputs
        del logits
        # torch.cuda.empty_cache()

        gen_scaling_factor = 1.0
        gen_scaling_factor_without_spec_tokens = 1.0
        if self.is_ddp:
            # gather num_elements for scaling results
            avg_gen_num_elements = self._dist_gather_scalar_mean(gen_num_elements)
            avg_gen_num_elements_without_spec_tokens = self._dist_gather_scalar_mean(gen_num_elements_without_spec_tokens)
            
            gen_scaling_factor = gen_num_elements / avg_gen_num_elements if avg_gen_num_elements > 0 else 0
            gen_scaling_factor_without_spec_tokens = gen_num_elements_without_spec_tokens / avg_gen_num_elements_without_spec_tokens \
                if avg_gen_num_elements_without_spec_tokens > 0 else 0
            
        # combine losses
        total_loss = gen_loss*gen_scaling_factor

        # TODO: for debug
        if total_loss.isnan() or total_loss.isinf():
            logger.error(f"total loss: {total_loss.item()}, gen_loss: {gen_loss.item()}")
            logger.error(f"gen_num_elements: {gen_num_elements}, gen_num_elements_without_spec_tokens: {gen_num_elements_without_spec_tokens}")
            logger.error(f"avg_gen_num_elements: {avg_gen_num_elements}, avg_gen_num_elements_without_spec_tokens: {avg_gen_num_elements_without_spec_tokens}")
            raise ValueError(f"Total loss is NaN or Inf: {total_loss.item()}")

        return TrainingOutput(
            loss=total_loss,
            gen_loss=gen_loss*gen_scaling_factor,
            gen_accuracy=gen_accuracy*gen_scaling_factor,
            gen_accuracy_without_spec_tokens=gen_accuracy_without_spec_tokens*gen_scaling_factor_without_spec_tokens,
        )

    def _dist_gather_scalar_mean(self, value: torch.Tensor):
        value_sum = value.clone()
        dist.all_reduce(value_sum, op=dist.ReduceOp.SUM)
        return value_sum.item() / self.world_size


    def reset_prefix_cache(self):
        if self.use_vllm:
            self.base_model.reset_prefix_cache()


    # inference
    def get_subquery(self, text):
        pattern = re.compile(f"{self.spec_tokens['bosq_token']}(.*?){self.spec_tokens['eosq_token']}", re.DOTALL)
        # pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
        else:
            return None
        
    def get_thoughts(self, text):
        pattern = re.compile(f"{self.spec_tokens['bot_token']}(.*?){self.spec_tokens['eot_token']}", re.DOTALL)
        # pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        matches = [item.strip() for item in matches if item.strip() != '']
        if matches and len(matches) > 0:
            # save all thoughts (sometimes the model generates multiple <think></think> thoughts)
            return ' '.join(matches)
        else:
            return None
        
    def get_answer(self, text):
        if self.spec_tokens['boa_token'] in text:
            answer = text.split(self.spec_tokens['boa_token'])[-1]
            if self.spec_tokens['eoa_token'] in answer:
                answer = answer.split(self.spec_tokens['eoa_token'])[0]

            return answer.strip()
        else:
            return None

    @torch.inference_mode()
    def _text_generation(self, all_prompts, eos_token_id=None):
        metadata = {'num_request': len(all_prompts)}
        
        eos_token_id_list = [self.tokenizer.eos_token_id]
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id_list.extend(eos_token_id)
        eos_token_id_list = list(set(eos_token_id_list))  # Remove duplicates

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
            eos_token_id=eos_token_id_list,
            pad_token_id=self.tokenizer.pad_token_id,
            # temperature=None, top_p=None, top_k=None,
        )

        outputs = outputs[:, input_batch['input_ids'].shape[1]:]  # Remove the input part
        metadata['completion_tokens'] = (outputs != self.tokenizer.pad_token_id).sum().item()
        all_responses = [self.tokenizer.decode(out, skip_special_tokens=False) for out in outputs]
        
        return all_responses, metadata

    def _text_generation_vllm(self, all_prompts, stopping_sequences=None):
        all_responses, metadata = self.base_model.batch_infer_with_prompt_texts(
            all_prompts, max_tokens=self.max_gen_tokens, 
            # temperature=0.85, top_p=0.95, 
            # presence_penalty=1.1,
            stopping_sequences=stopping_sequences)

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
            assert not any(item.chunk_items for item in infer_inputs), "SFTModel should not have any retrieved chunks in the first turn."
            
            # Initial prompt
            all_prompts = [self.spec_prompts['query_prompt'].format(query=item.query) for item in infer_inputs]
            if self.use_chat_template:
                all_prompts = self.tokenizer.apply_chat_template(
                    [[{"role": "user", "content": p}] for p in all_prompts], 
                    add_generation_prompt=True, tokenize=False)
        else:
            all_prompts = past_prompts
            search_texts = [
                self.spec_tokens['bod_token'] + 
                self._process_chunk_texts([chunk_item.text for chunk_item in item.chunk_items[-1]], 
                                          title_in_first_line=self.title_in_first_line) + 
                self.spec_tokens['eod_token'] for item in infer_inputs
            ]
            all_prompts = [prompt + search_text for prompt, search_text in zip(all_prompts, search_texts)]

        all_prompts = [prompt + self.spec_tokens['bot_token'] for prompt in all_prompts]

        if self.use_vllm:
            all_think_responses, metadata = self._text_generation_vllm(all_prompts, stopping_sequences=[self.spec_tokens['eot_token']])
        else: 
            all_think_responses, metadata = self._text_generation(all_prompts, eos_token_id=self.spec_token_ids['eot_token'])

        all_think_responses = [out if out.endswith(self.spec_tokens['eot_token']) 
                               else out + '...(truncated)\n' + self.spec_tokens['eot_token'] 
                               for out in all_think_responses]
        all_prompts = [prompt + think_response for prompt, think_response in zip(all_prompts, all_think_responses)]
        if self.use_vllm:
            all_subquery_responses, metadata = self._text_generation_vllm(all_prompts, stopping_sequences=[self.spec_tokens['eosq_token'], self.spec_tokens['eoa_token']])
        else:
            all_subquery_responses, metadata = self._text_generation(all_prompts, eos_token_id=[self.spec_token_ids['eosq_token'], self.spec_token_ids['eoa_token']])
        all_responses = [self.spec_tokens['bot_token'] + think_response + response for think_response, response in zip(all_think_responses, all_subquery_responses)]

        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            all_record_responses = [[out] for out in all_responses]

        past_prompts = [prompt + output for prompt, output in zip(all_prompts, all_subquery_responses)]
        all_new_thoughts = [self.get_thoughts(output) for output in all_responses]
        all_answers = [self.get_answer(output) for output in all_responses]
        # if an answer is provided, then no additional subqueries
        all_new_subqueries = [self.get_subquery(output) if all_answers[i] is None else None for i, output in enumerate(all_responses)]
        # if no additional subquery is provided and no answer, then set the answer to "" (fail)
        # all_answers = ["" if ans is None and all_new_subqueries[i] is None else ans for i, ans in enumerate(all_answers)]
        all_new_subqueries = ["" if subquery is None and all_answers[i] is None else subquery for i, subquery in enumerate(all_new_subqueries)]
        all_force_answer_flags = [False for _ in range(batch_size)]
        
        force_answer = cur_turn >= self.max_turns
        if force_answer:
            unfinished_prompts = {i: past_prompts[i] for i, ans in enumerate(all_answers) if ans is None or ans == ""}
            if len(unfinished_prompts) > 0:
                for i in unfinished_prompts:
                    if unfinished_prompts[i].endswith(self.spec_tokens['eosq_token']):
                        unfinished_prompts[i] = unfinished_prompts[i].rsplit(self.spec_tokens['bosq_token'], 1)[0]
                    # force answer
                    unfinished_prompts[i] += self.spec_tokens['boa_token']

                if self.use_vllm:
                    unfinished_all_responses, unfinished_metadata = self._text_generation_vllm(list(unfinished_prompts.values()), stopping_sequences=[self.spec_tokens['eoa_token']])
                else:
                    unfinished_all_responses, unfinished_metadata = self._text_generation(list(unfinished_prompts.values()), eos_token_id=self.spec_token_ids['eoa_token'])
                
                if os.getenv("DEBUG_MODE", "false").lower() == "true":
                    unfinished_all_record_responses = [[out] for out in unfinished_all_responses]
                
                unfinished_all_responses = [self.spec_tokens['boa_token'] + out for out in unfinished_all_responses]
                unfinished_all_responses = [out.split(self.spec_tokens['eoa_token'])[0] + self.spec_tokens['eoa_token'] if self.spec_tokens['eoa_token'] in out else out + self.spec_tokens['eoa_token']
                                            for out in unfinished_all_responses]
                unfinished_all_answers = [self.get_answer(output) for output in unfinished_all_responses]
                # force finish
                unfinished_all_answers = [ans if ans is not None else "" for ans in unfinished_all_answers]

                for cur_idx, global_idx in enumerate(unfinished_prompts.keys()):
                    all_answers[global_idx] = unfinished_all_answers[cur_idx]
                    all_new_subqueries[global_idx] = None
                    past_prompts[global_idx] = unfinished_prompts[global_idx].removesuffix(self.spec_tokens['boa_token']) + unfinished_all_responses[cur_idx]
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


    # inference with kv cache
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
            assert not infer_inputs.chunk_items, "SFTModel should not have any retrieved chunks in the first turn."
            
            # prompt = self.spec_tokens['boq_token'] + infer_inputs.query + self.spec_tokens['eoq_token']
            prompt = self.spec_prompts['query_prompt'].format(query=infer_inputs.query)
            if self.use_chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], 
                    add_generation_prompt=True, tokenize=False)
        else: 
            prompt = self.spec_tokens['bod_token']
            prompt += self._process_chunk_texts([chunk_item.text for chunk_item in infer_inputs.chunk_items[-1]], 
                                                title_in_first_line=self.title_in_first_line)
            prompt += self.spec_tokens['eod_token']
        
        prefill_start_time = time.perf_counter()
        _, past_key_values, metadata = self._get_forward_hidden_state_outputs(prompt, past_key_values)  # update past_key_values with the retrieved chunks
        total_metadata['prompt_tokens'] += metadata['prompt_tokens']
        total_metadata['completion_tokens'] += metadata['completion_tokens']
        prompt = ''
        prefill_end_time = time.perf_counter()
        prefill_total_time = prefill_end_time - prefill_start_time
        
        thought_generation_start_time = time.perf_counter()
        
        prompt += self.spec_tokens['bot_token']
        thought, _, past_key_values, metadata = self._text_generation_with_kv_cache(
            prompt, past_key_values, end_token_id=self.spec_token_ids['eot_token'], 
            max_gen_tokens=self.max_gen_tokens
        )
        prompt = self.spec_tokens['eot_token']
        infer_outputs.thoughts.append(thought.strip())
        total_metadata['prompt_tokens'] += metadata['prompt_tokens']
        total_metadata['completion_tokens'] += metadata['completion_tokens']
        next_token_id, _, past_key_values, metadata = self._next_token_id_choice(
            prompt, past_key_values, 
            candidate_token_ids=[self.spec_token_ids['bosq_token'], self.spec_token_ids['boa_token']]
        )
        prompt = ''
        
        thought_generation_end_time = time.perf_counter()
        thought_generation_total_time = thought_generation_end_time - thought_generation_start_time

        if cur_turn >= self.max_turns:
            next_token_id = self.spec_token_ids['boa_token']
            infer_outputs.force_answer = True

        if next_token_id == self.spec_token_ids['boa_token']:
            answer_generation_start_time = time.perf_counter()
            
            prompt += self.spec_tokens['boa_token']
            answer, _, past_key_values, metadata = self._text_generation_with_kv_cache(
                prompt, past_key_values, end_token_id=self.spec_token_ids['eoa_token'], 
                max_gen_tokens=self.max_gen_tokens
            )
            infer_outputs.answer = answer.strip()
            total_metadata['prompt_tokens'] += metadata['prompt_tokens']
            total_metadata['completion_tokens'] += metadata['completion_tokens']
            
            answer_generation_end_time = time.perf_counter()
            answer_generation_total_time = answer_generation_end_time - answer_generation_start_time
            subquery_generation_total_time = None

            infer_outputs.states = None
        else: 
            subquery_generation_start_time = time.perf_counter()

            prompt += self.spec_tokens['bosq_token']
            subquery, _, past_key_values, metadata = self._text_generation_with_kv_cache(
                prompt, past_key_values, end_token_id=self.spec_token_ids['eosq_token'], 
                max_gen_tokens=self.max_gen_tokens, input_end_token=True
            )
            infer_outputs.subqueries.append(subquery.strip())
            total_metadata['prompt_tokens'] += metadata['prompt_tokens']
            total_metadata['completion_tokens'] += metadata['completion_tokens']
            
            subquery_generation_end_time = time.perf_counter()
            subquery_generation_total_time = subquery_generation_end_time - subquery_generation_start_time
            answer_generation_total_time = None
                
            infer_outputs.states = {
                "past_key_values": past_key_values,
            }

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

    def _process_chunk_texts(self, texts, title_in_first_line=True): 
        format_reference = ''
        for idx, text in enumerate(texts):
            if title_in_first_line:
                title = text.split("\n")[0]
                text = "\n".join(text.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            else:
                format_reference += f"[Doc {idx+1}]\n{text}\n"

        return format_reference

    @torch.inference_mode()
    def _get_forward_hidden_state_outputs(self, input_data, past_key_values):
        input_token_ids = self.tokenizer.encode(input_data, return_tensors='pt', add_special_tokens=False).to(self.base_model.device)

        metadata = dict()
        metadata['prompt_tokens'] = input_token_ids.size(1)
        metadata['completion_tokens'] = 0
            
        outputs = self.base_model(
            input_ids=input_token_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-1]
        past_key_values = outputs.past_key_values

        return last_hidden_state, past_key_values, metadata

    @torch.inference_mode()
    def _next_token_id_choice(self, input_data, past_key_values, candidate_token_ids, output_full_hidden_state=False):
        input_token_ids = self.tokenizer.encode(input_data, return_tensors='pt', add_special_tokens=False).to(self.base_model.device)
            
        metadata = dict()
        metadata['prompt_tokens'] = input_token_ids.size(1)
        metadata['completion_tokens'] = 1

        outputs = self.base_model(
            input_ids=input_token_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        logits = outputs.logits[:, -1, :]
        candidate_logits = logits[:, candidate_token_ids]
        best_idx = candidate_logits.argmax(dim=-1).item()
        next_token_id = candidate_token_ids[best_idx]
        full_hidden_state = outputs.hidden_states
        last_hidden_state = outputs.hidden_states[-1]
        past_key_values = outputs.past_key_values

        if output_full_hidden_state:
            return next_token_id, full_hidden_state, past_key_values, metadata
        
        return next_token_id, last_hidden_state, past_key_values, metadata

    @torch.inference_mode()
    def _text_generation_with_kv_cache(self, input_data, past_key_values, end_token_id=None,
                                       max_gen_tokens=64, input_end_token=False, 
                                       output_full_hidden_state=False):
        input_token_ids = self.tokenizer.encode(input_data, return_tensors='pt', add_special_tokens=False).to(self.base_model.device)

        metadata = dict()
        metadata['prompt_tokens'] = input_token_ids.size(1)
        metadata['completion_tokens'] = 0

        outputs = self.base_model(
            input_ids=input_token_ids,
            past_key_values=past_key_values, 
            output_hidden_states=True,
        )
        
        output_token_ids = []
        input_last_hidden_state = outputs.hidden_states[-1] # without the generation tokens
        input_full_hidden_state = outputs.hidden_states
        past_key_values = outputs.past_key_values
        input_token_ids = torch.argmax(outputs.logits[0:1, -1:, :], dim=-1,keepdim=False)
        token_id = input_token_ids.item()
        output_token_ids.append(token_id)
        
        while (token_id is None or end_token_id is None or token_id != end_token_id) and (len(output_token_ids) < max_gen_tokens):
            outputs = self.base_model(
                input_ids=input_token_ids,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            input_token_ids = torch.argmax(outputs.logits[0:1, -1:, :], dim=-1,keepdim=False)
            token_id = input_token_ids.item()
            output_token_ids.append(token_id)

        num_gen_tokens = len(output_token_ids)
        output_text = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

        if input_end_token: 
            outputs = self.base_model(
                input_ids=torch.tensor([[end_token_id]], device=self.base_model.device),
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            num_gen_tokens += 1
            
        metadata['completion_tokens'] = num_gen_tokens

        if output_full_hidden_state:
            return output_text, input_full_hidden_state, past_key_values, metadata
        
        return output_text, input_last_hidden_state, past_key_values, metadata

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable()
        # if hasattr(self, 'encoder') and self.encoder is not None:
        #     self.encoder.gradient_checkpointing_enable(**kwargs)

    @classmethod
    def get_special_added_tokens(cls, config): 
        spec_tokens = dict()

        # begin and end of decode thought/subquery
        spec_tokens['bot_token'] = '<thinking>'
        spec_tokens['eot_token'] = '</thinking>'
        spec_tokens['bosq_token'] = '<subquery>'
        spec_tokens['eosq_token'] = '</subquery>'

        # tokens for retrieved docs
        spec_tokens['bod_token'] = '<information>'
        spec_tokens['eod_token'] = '</information>'
        
        # tokens for final answer
        spec_tokens['boa_token'] = '<answer>'
        spec_tokens['eoa_token'] = '</answer>'

        # create a list of all special tokens
        spec_token_list = sorted({
            t for v in spec_tokens.values() for t in (v if isinstance(v, list) else [v])
        })

        return spec_tokens, spec_token_list
    
    @classmethod
    def get_special_prompts(cls, config): 
        spec_tokens, _ = cls.get_special_added_tokens(config)
        spec_prompts = dict()
        spec_prompts['query_prompt'] = 'Answer the following question by reasoning step by step and retrieving necessary information at each step:\n' + "{query}\n"

        return spec_prompts
        

    @classmethod
    def _resize_model_token_embeddings(cls, model, tokenizer, config): 
        model.resize_token_embeddings(len(tokenizer))
        base_embedding_layer = model.get_input_embeddings()
        base_lm_head = model.get_output_embeddings() # The input embeddings and lm heads are tied in some models, so maybe unnecessary
        target_id = tokenizer.convert_tokens_to_ids("<<")
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        _, spec_token_list = cls.get_special_added_tokens(config)
        spec_token_id_list = [tokenizer.convert_tokens_to_ids(token) for token in spec_token_list]
        for token_id in spec_token_id_list:
            base_embedding_layer.weight.data[token_id] = base_embedding_layer.weight.data[target_id]
            base_lm_head.weight.data[token_id] = base_lm_head.weight.data[target_id]

    @classmethod
    def build(cls, config):
        tokenizer = load_tokenizer_and_add_special_tokens(
            config=config
        )
        
        torch_dtype = torch.float32
        if getattr(config.training, 'bf16', False): 
            torch_dtype = torch.bfloat16
        elif getattr(config.training, 'fp16', False):
            torch_dtype = torch.float16
        
        base_model = AutoModelForCausalLMSafeLoader.from_pretrained(
            config.model.model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=config.model.attn_implementation
        )
            
        cls._resize_model_token_embeddings(base_model, tokenizer, config)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        base_model.config.use_cache = config.model.use_cache
        if config.model.lora.use_lora:
            if config.training.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if config.model.lora.lora_name_or_path:
                lora_model = PeftModel.from_pretrained(base_model, config.model.lora.lora_name_or_path, is_trainable=True)
                logger.info(f"Loaded LoRA weights from {config.model.lora.lora_name_or_path}")
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=config.model.model_name_or_path,
                    task_type=TaskType.CAUSAL_LM,
                    r=config.model.lora.lora_r,
                    lora_alpha=config.model.lora.lora_alpha,
                    lora_dropout=config.model.lora.lora_dropout,
                    target_modules=config.model.lora.lora_target_modules.split(','),
                    inference_mode=False,
                    use_rslora=config.model.lora.use_rslora,
                )
                lora_model = get_peft_model(base_model, lora_config)

            for param in lora_model.get_input_embeddings().parameters():
                param.requires_grad = True
            for param in lora_model.get_output_embeddings().parameters():
                param.requires_grad = True

            logger.info("LLM model trainable parameters:")
            lora_model.print_trainable_parameters()

            base_model = lora_model

        model = cls(
            base_model=base_model,
            tokenizer=tokenizer,
            config=config,
        )
            
        return model
    
    @classmethod
    def load(cls, config):
        torch_dtype = torch.float32
        if getattr(config.training, 'bf16', False): 
            torch_dtype = torch.bfloat16
        elif getattr(config.training, 'fp16', False):
            torch_dtype = torch.float16
        
        if config.vllm.use_vllm:
            tokenizer = load_tokenizer(tokenizer_name_or_path=config.model.model_name_or_path)
            base_model = VLLMOffline(config.model.model_name_or_path, seed=config.training.seed, 
                                     gpu_memory_utilization=config.vllm.qa_model_gpu_memory_utilization,
                                     max_model_len=config.vllm.qa_model_max_model_len,
                                     max_num_seqs=config.vllm.qa_model_max_num_seqs,
                                     dtype=torch_dtype)
        else:
            tokenizer = load_tokenizer_and_add_special_tokens(
                config=config
            )
            base_model = AutoModelForCausalLMSafeLoader.from_pretrained(
                config.model.model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=config.model.attn_implementation, 
                device_map="auto"
            )
            cls._resize_model_token_embeddings(base_model, tokenizer, config)
            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = tokenizer.pad_token_id

            if config.model.lora.lora_name_or_path:
                lora_config = LoraConfigSafeLoader.from_pretrained(
                    config.model.lora.lora_name_or_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=config.model.attn_implementation,)
                lora_model = PeftModel.from_pretrained(base_model, config.model.lora.lora_name_or_path, config=lora_config, device_map="auto")
                lora_model = lora_model.merge_and_unload()
                logger.info(f"Loaded and merged LoRA weights from {config.model.lora.lora_name_or_path}")

                base_model = lora_model
            
        model = cls(
            base_model=base_model,
            tokenizer=tokenizer,
            config=config
        )

        return model
    
    def save(self, output_dir: str, state_dict=None, save_safetensors: bool=True):
        if state_dict is None:
            state_dict = self.state_dict()
        base_prefix = 'base_model.'
        # assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        base_state_dict = {k[len(base_prefix):]: v for k, v in state_dict.items() if k.startswith(base_prefix)}
        self.base_model.save_pretrained(
            output_dir, state_dict=base_state_dict, safe_serialization=save_safetensors, 
            save_embedding_layers=True # save the input and output embeddings due to the added special tokens
        )       
