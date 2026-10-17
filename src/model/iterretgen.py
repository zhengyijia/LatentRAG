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

class IterRetGen(nn.Module):
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
        cur_turn = len(infer_inputs[0].chunk_items)-1
        logger.info(f"Current turn: {cur_turn}, batch size: {batch_size}")

        search_texts = [
            process_chunk_texts([chunk_item.text for chunk_item in item.chunk_items[-1]], 
                                title_in_first_line=self.title_in_first_line) for item in infer_inputs
        ]
        prompt = (
            "Answer the question based on the given document."
            "Only give me the answer and do not output any other words."
            "\nThe following are given documents.\n\n{search_results}"
            "\n\nQuestion: {question}\n"
        )
        all_prompts = [prompt.format(search_results=search_text, question=item.query) for search_text, item in zip(search_texts, infer_inputs)]
        if self.use_chat_template:
            all_prompts = self.tokenizer.apply_chat_template(
                [[{"role": "user", "content": p}] for p in all_prompts], 
                add_generation_prompt=True, tokenize=False)

        if self.use_vllm:
            all_responses, metadata = self._text_generation_vllm(all_prompts)
        else: 
            all_responses, metadata = self._text_generation(all_prompts)

        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            all_record_responses = [[out] for out in all_responses]

        all_answers = all_responses if cur_turn == self.max_turns - 1 else [None for _ in range(batch_size)]
        all_new_subqueries = [f"{item.query} {out}" for out, item in zip(all_responses, infer_inputs)] if cur_turn < self.max_turns - 1 else [None for _ in range(batch_size)]

        infer_outputs = deepcopy(infer_inputs)
        for i, item in enumerate(infer_outputs):
            item.subqueries.append(all_new_subqueries[i])
            item.answer = all_answers[i]
            
            if os.getenv("DEBUG_MODE", "false").lower() == "true":
                record_responses = item.states.get("record_responses", [])
                record_responses.extend(all_record_responses[i])
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
