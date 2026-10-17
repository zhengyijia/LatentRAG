import re
import json
import logging
import time
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Optional, List
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from llm_tools.vllm_offline import VLLMOffline
from model.infer_inputs_outputs import IterationRecord
from llm_tools.safe_loader import AutoModelForCausalLMSafeLoader, AutoTokenizerSafeLoader

logger = logging.getLogger(__name__)

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

class Few_Shot_RAG(nn.Module):

    def __init__(self, 
                 base_model: VLLMOffline,
                 tokenizer: PreTrainedTokenizer,
                 max_retrieved_docs: int,
                 use_chat_template: bool,
                 max_gen_tokens: int, 
                 use_vllm: bool, 
                 few_shot_samples: Optional[List[Dict]], 
                 config):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_retrieved_docs = max_retrieved_docs
        self.use_chat_template = use_chat_template
        self.max_gen_tokens = max_gen_tokens
        self.use_vllm = use_vllm
        self.qa_answer_prefix = 'The answer is:'
        self.config = config

        self.few_shot_samples = few_shot_samples if few_shot_samples is not None else []
        # self.few_shot_samples = self.few_shot_samples[:20]  # use only at most first 20 samples for few-shot learning
        self.few_shot_samples = self.few_shot_samples[:1]  # use only one-shot example for few-shot learning
        self.few_shot_samples = [self._render_sample_prompts(sample) for sample in self.few_shot_samples]
        if self.use_chat_template:
            self.common_prompt_token_ids = self._prepare_common_prompt_token_ids_use_chat_template()
            default_system_prompt = self.tokenizer.apply_chat_template( # used later to remove the default system prompt in Llama
                [{"role": "system", "content": ""}],
                add_generation_prompt=False,
                tokenize=True,
                add_special_tokens=False,
            )
            self.len_llm_default_system_prompt = len(default_system_prompt)
        else:
            self.common_prompt_token_ids = self._prepare_common_prompt_token_ids()
            
        if not use_vllm:
            self.stop_token_ids = self._get_stop_token_ids()
        else:
            self.stopping_sequences = ["\n"]
            self.stop_token_ids = self._get_stop_token_ids()
            
    def reset_prefix_cache(self):
        if self.use_vllm:
            self.base_model.reset_prefix_cache()
            
    def _get_stop_token_ids(self):
        tokenizer = self.tokenizer
        vocab_size = tokenizer.vocab_size
        stop_token_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
        for token_id in range(vocab_size):
            try:
                token = tokenizer.decode([token_id])
                if '\n' in token:
                    stop_token_ids.add(token_id)
            except:
                continue

        return list(stop_token_ids)

    def get_tokenizer(self):
        return self.tokenizer
    
    def _render_sample_prompts(self, sample): 
        user_prompt = ''

        # if max_retrieved_docs is 0, reduce to traditional CoT prompt. Otherwise, use IRCoT prompt
        if self.max_retrieved_docs > 0:
            for chunk in sample['retrieved']:
                # add quotes around the title
                chunk = chunk.split('\n', 1)
                chunk[0] = f'"{chunk[0]}"'
                chunk = chunk[0] + '\n' + chunk[1]
                
                user_prompt += f'Wikipedia Title: {chunk}\n\n'
        user_prompt += f'Query: Answer the following question. Only provide the final answer without any explanation or extra text.\n{sample["question"]}'

        assistant_prompt = f'{self.qa_answer_prefix} {sample["answer"]}'

        return {
            'user_prompt': user_prompt,
            'assistant_prompt': assistant_prompt
        }
        
    def _prepare_common_prompt_token_ids_use_chat_template(self): 
        chat_data = [
            # {"role": "system", "content": self.ircot_reason_instruction}, # without system prompt
        ]
        for sample in self.few_shot_samples:
            chat_data.append({"role": "user", "content": sample['user_prompt']})
            chat_data.append({"role": "assistant", "content": sample['assistant_prompt']})
        
        prompt_token_ids = self.tokenizer.apply_chat_template(
            chat_data,
            add_generation_prompt=False,
            tokenize=True,
            add_special_tokens=False,
        )
        
        return prompt_token_ids
    
    def _prepare_common_prompt_token_ids(self):
        # input_text = system_prompt + '\n\n\n'
        input_text = ''  # without system prompt
        for sample in self.few_shot_samples:
            input_text += f'{sample["user_prompt"]}\n\n{sample["assistant_prompt"]}\n\n\n'
            
        prompt_token_ids = self.tokenizer(
            input_text, add_special_tokens=False, return_attention_mask=False).input_ids
        
        return prompt_token_ids
    
    def _prepare_input_text_use_chat_template(self, user_prompt, assistant_prompt):
        chat_data = [{"role": "system", "content": ""}] # used to get default system prompt in Llama, will remove later
        chat_data.append({"role": "user", "content": user_prompt})
        
        input_text = self.tokenizer.apply_chat_template(
            chat_data,
            add_generation_prompt=True,
            tokenize=False,
            add_special_tokens=False,
        )
        input_text += assistant_prompt

        return input_text

    def _prepare_input_text(self, user_prompt, assistant_prompt):
        input_text = ''  # without system prompt
        input_text += f'{user_prompt}\n\n{assistant_prompt}'

        return input_text

    def _render_prompts_and_construct_input_token_ids(self, all_queries, all_retrieved_chunks, force_answer=False):
        all_user_input_texts_with_assistant_prefix = []
        for query, retrieved_chunks in zip(all_queries, all_retrieved_chunks):
            user_prompt = ''
            for chunk in retrieved_chunks:
                user_prompt += f'Wikipedia Title: {chunk}\n\n'
            user_prompt += f'Query: Answer the following question. Only provide the final answer without any explanation or extra text.\n{query}'

            assistant_prompt = ""
            if force_answer:
                assistant_prompt += f'{self.qa_answer_prefix} '
            
            if self.use_chat_template:
                input_text = self._prepare_input_text_use_chat_template(user_prompt, assistant_prompt)
            else:
                input_text = self._prepare_input_text(user_prompt, assistant_prompt)

            all_user_input_texts_with_assistant_prefix.append(input_text)

        all_user_prompt_token_ids_with_assistant_prefix = self.tokenizer(
            all_user_input_texts_with_assistant_prefix, 
            add_special_tokens=False, 
            return_attention_mask=False,
            padding=False,
            truncation=False, 
        ).input_ids
        if self.use_chat_template:
            all_user_prompt_token_ids_with_assistant_prefix = [
                prompt_token_ids[self.len_llm_default_system_prompt:]  # remove the default system prompt in Llama
                for prompt_token_ids in all_user_prompt_token_ids_with_assistant_prefix
            ]
            
        all_prompt_token_ids = [
            self.common_prompt_token_ids + prompt_token_ids
            for prompt_token_ids in all_user_prompt_token_ids_with_assistant_prefix
        ]
        
        return all_prompt_token_ids

    def _text_generation(self, all_prompt_token_ids, stop_token_ids=None):
        metadata = {'num_request': len(all_prompt_token_ids)}
        metadata['prompt_tokens'] = sum([len(all_prompt_token_ids[i]) for i in range(len(all_prompt_token_ids))])
        
        input_batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            {"input_ids": all_prompt_token_ids},
            padding=True,
            return_tensors='pt',
        ).to(self.base_model.device)
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                **input_batch,
                max_new_tokens=self.max_gen_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=stop_token_ids, 
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=None, top_p=None, top_k=None,
            )
            
        outputs = outputs[:, input_batch['input_ids'].shape[1]:]  # Remove the input part
        metadata['completion_tokens'] = (outputs != self.tokenizer.pad_token_id).sum().item()
        all_responses = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        
        return all_responses, metadata

    
    def _text_generation_vllm(self, all_prompt_token_ids, stopping_sequences=None):
        all_responses, metadata = self.base_model.batch_infer_with_prompt_token_ids(
                prompt_token_ids=all_prompt_token_ids, 
                max_tokens=self.max_gen_tokens,
                stopping_sequences=stopping_sequences
        )

        return all_responses, metadata

    def batch_generate(self, infer_inputs: List[IterationRecord]) -> List[IterationRecord]:
        start_time = time.perf_counter()
        
        batch_size = len(infer_inputs)
        logger.info(f"Batch size: {batch_size}")
        
        all_queries = [item.query for item in infer_inputs]
        all_retrieved_chunks = [[chunk_item.text for chunk_item in item.chunk_items[0]]
                                for item in infer_inputs]
        all_retrieved_chunks = [chunks[:self.max_retrieved_docs] for chunks in all_retrieved_chunks]

        all_prompt_token_ids = self._render_prompts_and_construct_input_token_ids(
            all_queries=all_queries,
            all_retrieved_chunks=all_retrieved_chunks,
            force_answer=True
        )
        
        if self.use_vllm:
            all_responses, metadata = self._text_generation_vllm(
                all_prompt_token_ids=all_prompt_token_ids,
                stopping_sequences=self.stopping_sequences
            )
        else: 
            all_responses, metadata = self._text_generation(
                all_prompt_token_ids=all_prompt_token_ids,
                stop_token_ids=self.stop_token_ids
            )

        all_answers = [response.split('\n')[0].strip() for response in all_responses]
        
        infer_outputs = deepcopy(infer_inputs)
        for i, item in enumerate(infer_outputs):
            item.answer = all_answers[i]
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / batch_size
        for item in infer_outputs:
            item.infer_time_each_steps.append(total_time)
            
        return infer_outputs, metadata
    
    @torch.inference_mode()
    def _text_generation_with_kv_cache(self, prompt, past_key_values=None, max_gen_tokens=64, stop_token_ids=[], input_end_token=False):        
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

        stop_flag = False
        if all_token_ids[-1].item() == self.tokenizer.eos_token_id:
            stop_flag = True
        elif all_token_ids[-1].item() in stop_token_ids:
            stop_flag = True
        
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
            elif all_token_ids[-1].item() in stop_token_ids:
                stop_flag = True

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
    def _forward_token_ids_with_kv_cache(self, token_ids, past_key_values):
            
        metadata = {'completion_tokens': 1}
        metadata['prompt_tokens'] = len(token_ids)
            
        outputs = self.base_model(
            input_ids=torch.tensor(token_ids, device=self.base_model.device, dtype=torch.long).unsqueeze(0),
            past_key_values=past_key_values,
        )

        past_key_values = outputs.past_key_values
        end_token_id = torch.argmax(outputs.logits[0:1, -1:, :], dim=-1,keepdim=False)
        end_token = self.tokenizer.decode([end_token_id.item()], skip_special_tokens=False)

        return past_key_values, metadata, end_token
    
    def generate_with_step_wise_time(self, infer_inputs: List[IterationRecord]) -> List[IterationRecord]:
        assert len(infer_inputs) == 1, "Step-wise time measurement only supports batch size 1."
        assert not self.use_vllm, "Step-wise time measurement does not support vLLM."
        
        infer_inputs = infer_inputs[0]
        
        start_time = time.perf_counter()
        
        query = infer_inputs.query
        retrieved_chunks = [chunk_item.text for chunk_item in infer_inputs.chunk_items[0]]
        retrieved_chunks = retrieved_chunks[:self.max_retrieved_docs]

        past_key_values = None

        prefill_start_time = time.perf_counter()
        prompt_token_ids = self._render_prompts_and_construct_input_token_ids(
            all_queries=[query],
            all_retrieved_chunks=[retrieved_chunks],
            force_answer=True
        )[0]
        past_key_values, metadata, end_token = self._forward_token_ids_with_kv_cache(prompt_token_ids, past_key_values)
        prefill_end_time = time.perf_counter()
        prefill_total_time = prefill_end_time - prefill_start_time

        answer_generation_start_time = time.perf_counter()
        response, past_key_values, metadata, end_token = self._text_generation_with_kv_cache(
            prompt=end_token, 
            past_key_values=past_key_values, 
            max_gen_tokens=self.max_gen_tokens,
            stop_token_ids=self.stop_token_ids)
        answer_generation_end_time = time.perf_counter()
        answer_generation_total_time = answer_generation_end_time - answer_generation_start_time

        answer = response.split('\n')[0].strip()
        
        infer_outputs = deepcopy(infer_inputs)
        infer_outputs.answer = answer
        
        if 'step_wise_time' not in infer_outputs.extra_info:
            infer_outputs.extra_info['step_wise_time'] = dict()
        if 'prefill_time' not in infer_outputs.extra_info['step_wise_time']:
            infer_outputs.extra_info['step_wise_time']['prefill_time'] = []
        if 'answer_generation_time' not in infer_outputs.extra_info['step_wise_time']:
            infer_outputs.extra_info['step_wise_time']['answer_generation_time'] = []
        infer_outputs.extra_info['step_wise_time']['prefill_time'].append(prefill_total_time)
        infer_outputs.extra_info['step_wise_time']['answer_generation_time'].append(answer_generation_total_time)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        infer_outputs.infer_time_each_steps.append(total_time)
            
        return [infer_outputs], metadata

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
        
        few_shot_samples_path = f"src/prompts/ircot_prompts/{config.data.few_shot_prompt_name}/gold_with_3_distractors_context_cot_qa_codex.json"
        with open(few_shot_samples_path, 'r') as f:
            few_shot_samples = json.load(f)
        
        model = cls(
            base_model=base_model,
            tokenizer=tokenizer,
            max_retrieved_docs=config.model.max_retrieved_docs, 
            use_chat_template=config.model.use_chat_template,
            max_gen_tokens=config.model.max_gen_tokens, 
            use_vllm=config.vllm.use_vllm, 
            few_shot_samples=few_shot_samples,
            config=config
        )

        return model
