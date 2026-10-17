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
import json5
from transformers import LogitsProcessor, LogitsProcessorList

from llm_tools.vllm_offline import VLLMOffline
from model.infer_inputs_outputs import IterationRecord
from llm_tools.safe_loader import AutoModelForCausalLMSafeLoader, AutoTokenizerSafeLoader

logger = logging.getLogger(__name__)


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        # input_ids: [batch, seq]
        for b in range(input_ids.shape[0]):
            unique_tokens = set(input_ids[b].tolist())
            scores[b, list(unique_tokens)] -= self.penalty
        return scores


def get_subquery(text):
    # pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    # match the last <tool_call>...</tool_call> in the text, and make sure no <tool_call> in between
    pattern = re.compile(r"<tool_call>((?:(?!<tool_call>|</tool_call>).)*)</tool_call>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        try:
            query = matches[-1].strip()
            query = json5.loads(query)
            if ('arguments' in query and 'query' in query['arguments']):
                if isinstance(query['arguments']['query'], str):
                    return query['arguments']['query'].strip()
                elif isinstance(query['arguments']['query'], list) and len(query['arguments']['query']) > 0 and isinstance(query['arguments']['query'][0], str):
                    return query['arguments']['query'][0].strip()
                else:
                    return None
            else:
                return None
        except Exception:
            logger.warning(f"Failed to parse subquery from tool_call: {matches[-1].strip()}")
            return None
    
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
    pattern = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        ans = matches[-1].strip()
        
        if not ans.endswith("</answer>"):
            ans += "...(truncated)"
        
        return ans
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

class AgentCPM(nn.Module):
    def __init__(self,
                 base_model: VLLMOffline,
                 tokenizer: PreTrainedTokenizer,
                 max_turns: int,
                 max_gen_tokens: int,
                 use_vllm: bool, 
                 config):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_gen_tokens = max_gen_tokens
        self.use_vllm = use_vllm
        self.config = config
        
        self.title_in_first_line = getattr(self.config.data, 'title_in_first_line', True)
        
        # self.answer_target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
            
        self.think_begin_token = '<think>'
        self.think_end_token = '</think>'
        self.think_end_token_id = self.tokenizer.convert_tokens_to_ids('</think>')
        
        self.logits_processor = LogitsProcessorList([
            PresencePenaltyLogitsProcessor(1.1)
        ])
        
        self.system_prompt = (
            "You are a deep research assistant. "
            "Your core function is to conduct thorough, multi-source investigations into any topic. "
            "You must handle both broad, open-domain inquiries and queries within specialized academic fields. "
            "For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. "
            "When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags."
            "\n\n# Tools\n\n"
            "You may call the search tool one or multiple times to assist with the user query.\n\n"
            "You are provided with the search tool, its signature within <tools></tools> XML tags:\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Perform Google web searches then returns a string of the top search results. Accepts a single query.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"The search query.\"}}, \"required\": [\"query\"]}}}"
            "\n</tools>\n\n"
            "For each search tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            "{\"name\": \"search\", \"arguments\": <args-json-object>}"
            "\n</tool_call>\n\n"
            "Current date: 2026-03-01"
        )
        # self.system_prompt = (
        #     "You are a helpful assistant."
        #     "\n\n# Tools\n\n"
        #     "You may call the search tool one or multiple times to assist with the user query.\n\n"
        #     "You are provided with the search tool, its signature within <tools></tools> XML tags:\n"
        #     "<tools>\n"
        #     "{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Perform Google web searches then returns a string of the top search results. Accepts a single query.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"The search query.\"}}, \"required\": [\"query\"]}}}"
        #     "\n</tools>\n\n"
        #     "For each search tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        #     "<tool_call>\n"
        #     "{\"name\": <function-name>, \"arguments\": <args-json-object>}"
        #     "\n</tool_call>\n"
        # )
        
        self.tool_call_failure_response = (
            "Search tool call failed, possibly due to an invalid tool-call format or excessively long generated output.\n"
            "NOTE: You are only allowed to call the search tool. Other tools are not available.\n"
            "NOTE: The search tool only accepts a single query. The query should be less than {self.max_gen_tokens} tokens.\n"
            "The signature of the search tool within <tools></tools> XML tags:\n"
            "<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"search\", \"description\": \"Perform Google web searches then returns a string of the top search results. Accepts a single query.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"The search query.\"}}, \"required\": [\"query\"]}}}"
            "\n</tools>\n"
        )
            
    def reset_prefix_cache(self):
        if self.use_vllm:
            self.base_model.reset_prefix_cache()

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
            temperature=None, top_p=None, top_k=None,
            logits_processor=self.logits_processor,
        )

        outputs = outputs[:, input_batch['input_ids'].shape[1]:]  # Remove the input part
        metadata['completion_tokens'] = (outputs != self.tokenizer.pad_token_id).sum().item()
        all_responses = [self.tokenizer.decode(out, skip_special_tokens=False) for out in outputs]
        
        return all_responses, metadata

    def _text_generation_vllm(self, all_prompts, stopping_sequences=None):
        all_responses, metadata = self.base_model.batch_infer_with_prompt_texts(
            all_prompts, max_tokens=self.max_gen_tokens, 
            # temperature=0.85, top_p=0.95, 
            presence_penalty=1.1,
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
            assert not any(item.chunk_items for item in infer_inputs), "AgentCPM should not have any retrieved chunks in the first turn."
            
            # Initial prompt
            all_messages = [[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": item.query}] for item in infer_inputs]
            all_prompts = self.tokenizer.apply_chat_template(all_messages, add_generation_prompt=True, tokenize=False)
            all_prompts = [prompt + self.think_begin_token for prompt in all_prompts]
        else:
            all_prompts = past_prompts
            search_texts = [
                process_chunk_texts([chunk_item.text for chunk_item in item.chunk_items[-1]], 
                                    title_in_first_line=self.title_in_first_line) for item in infer_inputs
            ]
            search_texts = [self.tool_call_failure_response if infer_inputs[i].subqueries[-1]=='' else search_text for i, search_text in enumerate(search_texts)]
            cur_search_template = '<tool_response>\n{search_results}\n</tool_response>'
            search_texts = [cur_search_template.format(search_results=text) for text in search_texts]
            search_text_messages = [[{"role": "user", "content": search_text}] for search_text in search_texts]
            search_text_prompts = self.tokenizer.apply_chat_template(search_text_messages, add_generation_prompt=True, tokenize=False)
            
            all_prompts = [prompt + '\n' + search_text_prompt + self.think_begin_token for prompt, search_text_prompt in zip(all_prompts, search_text_prompts)]

        if self.use_vllm:
            all_think_responses, metadata = self._text_generation_vllm(all_prompts, stopping_sequences=[self.think_end_token])
        else: 
            all_think_responses, metadata = self._text_generation(all_prompts, eos_token_id=self.think_end_token_id)
            
        all_think_responses = [out if out.endswith(self.think_end_token) 
                               else out + '...(truncated)\n' + self.think_end_token 
                               for out in all_think_responses] 
        # all_think_responses = [out.split(self.think_end_token)[0] + self.think_end_token
        #                         for out in all_think_responses]
        all_prompts = [prompt + think_response for prompt, think_response in zip(all_prompts, all_think_responses)]
        if self.use_vllm:
            all_subquery_responses, metadata = self._text_generation_vllm(all_prompts)
        else:
            all_subquery_responses, metadata = self._text_generation(all_prompts)
        all_responses = [self.think_begin_token + think_response + response for think_response, response in zip(all_think_responses, all_subquery_responses)]

        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            all_record_responses = [[out] for out in all_responses]

        past_prompts = [prompt + output for prompt, output in zip(all_prompts, all_subquery_responses)]
        all_new_thoughts = [get_thoughts(output) for output in all_responses]
        all_answers = [get_answer(output) for output in all_responses]
        # if an answer is provided, then no additional subqueries
        all_new_subqueries = [get_subquery(output) if all_answers[i] is None else None for i, output in enumerate(all_responses)]
        # if no additional subquery is provided and no answer, then set the answer to "" (fail)
        # all_answers = ["" if ans is None and all_new_subqueries[i] is None else ans for i, ans in enumerate(all_answers)]
        all_new_subqueries = ["" if subquery is None and all_answers[i] is None else subquery for i, subquery in enumerate(all_new_subqueries)]
        all_force_answer_flags = [False for _ in range(batch_size)]
        
        force_answer = cur_turn >= self.max_turns
        if force_answer:
            unfinished_prompts = {i: past_prompts[i] for i, ans in enumerate(all_answers) if ans is None or ans == ""}
            if len(unfinished_prompts) > 0:
                for i in unfinished_prompts:
                    # if unfinished_prompts[i].endswith("</tool_call>"):
                    #     unfinished_prompts[i] = unfinished_prompts[i].rsplit("<tool_call>", 1)[0]
                    unfinished_prompts[i] = unfinished_prompts[i].rsplit("<tool_call>", 1)[0]
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
    def _text_generation_with_kv_cache(self, prompt, past_key_values=None, max_gen_tokens=64, eos_token_id=None, input_end_token=False, 
                                       repetition_penalty=1.1):        
        input_token_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(self.base_model.device)
        
        metadata = dict()
        metadata['prompt_tokens'] = input_token_ids.size(1)
        metadata['completion_tokens'] = 0

        eos_token_id_list = [self.tokenizer.eos_token_id]
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id_list.extend(eos_token_id)
        eos_token_id_list = list(set(eos_token_id_list))  # Remove duplicates
        
        def apply_repetition_penalty(logits, generated_token_ids, penalty):
            # logits: [1, vocab]
            if penalty is None or penalty == 1.0:
                return logits

            for t in set(generated_token_ids):
                if t < 0 or t >= logits.size(-1):
                    continue
                if logits[0, t] > 0:
                    logits[0, t] /= penalty
                else:
                    logits[0, t] *= penalty
            return logits

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
        
        generated_token_ids = []
        generated_token_ids.append(token_id.item())
        
        stop_flag = False
        if all_token_ids[-1].item() in eos_token_id_list:
            stop_flag = True
        
        while not stop_flag and (all_token_ids.numel() < max_gen_tokens):
            outputs = self.base_model(
                input_ids=input_token_ids,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            logits = apply_repetition_penalty(
                logits, generated_token_ids, repetition_penalty
            )
            input_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = input_token_ids.view(-1).long()
            
            all_token_ids = torch.cat([all_token_ids, token_id], dim=0)
            generated_token_ids.append(token_id.item())
            
            if all_token_ids[-1].item() in eos_token_id_list:
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
            assert not infer_inputs.chunk_items, "AgentCPM should not have any retrieved chunks in the first turn."
            
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": infer_inputs.query}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else: 
            search_text = process_chunk_texts([chunk_item.text for chunk_item in infer_inputs.chunk_items[-1]], 
                                              title_in_first_line=self.title_in_first_line)
            search_text = self.tool_call_failure_response if infer_inputs.subqueries[-1]=='' else search_text
            cur_search_template = '<tool_response>\n{search_results}\n</tool_response>'
            search_text = cur_search_template.format(search_results=search_text)
            search_text_messages = [{"role": "user", "content": search_text}]
            search_text_prompt = self.tokenizer.apply_chat_template(search_text_messages, add_generation_prompt=True, tokenize=False)
            search_text_prompt = '\n' + search_text_prompt
            prompt = search_text_prompt
            
        prefill_start_time = time.perf_counter()
        past_key_values, prefill_metadata, end_token = self._forward_with_kv_cache(prompt, past_key_values)
        total_metadata['prompt_tokens'] += prefill_metadata['prompt_tokens']
        total_metadata['completion_tokens'] += prefill_metadata['completion_tokens']
        prefill_end_time = time.perf_counter()
        prefill_total_time = prefill_end_time - prefill_start_time
            
        thought_generation_start_time = time.perf_counter()
        # prompt = end_token
        prompt = self.think_begin_token
        think_response, past_key_values, think_metadata, end_token = self._text_generation_with_kv_cache(
            prompt=prompt, 
            past_key_values=past_key_values, 
            max_gen_tokens=self.max_gen_tokens,
            eos_token_id=self.think_end_token_id,)
        total_metadata['prompt_tokens'] += think_metadata['prompt_tokens']
        total_metadata['completion_tokens'] += think_metadata['completion_tokens']
        think_response = prompt + think_response
        thought_generation_end_time = time.perf_counter()
        thought_generation_total_time = thought_generation_end_time - thought_generation_start_time

        prompt = end_token
        if prompt != self.think_end_token:
            prompt = end_token + '...(truncated)\n' + self.think_end_token
            think_response += '...(truncated)\n' + self.think_end_token
        force_answer = cur_turn >= self.max_turns
        if not force_answer:
            subquery_generation_start_time = time.perf_counter()
            search_response, past_key_values, search_metadata, end_token = self._text_generation_with_kv_cache(
                prompt=prompt, 
                past_key_values=past_key_values, 
                max_gen_tokens=self.max_gen_tokens,
                eos_token_id=None,
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
            # answer = "" if answer is None and new_subquery is None else answer
            new_subquery = "" if new_subquery is None and answer is None else new_subquery
            force_answer_flag = False
            
            if answer is not None:
                answer_generation_total_time = subquery_generation_total_time
                subquery_generation_total_time = None
        else:
            answer_generation_start_time = time.perf_counter()
            
            prompt = prompt + "\n\n<answer>"
            answer_response, _, answer_metadata, _ = self._text_generation_with_kv_cache(
                prompt=prompt, 
                past_key_values=past_key_values,
                max_gen_tokens=self.max_gen_tokens,
                eos_token_id=None,
                input_end_token=True)
            
            total_metadata['prompt_tokens'] += answer_metadata['prompt_tokens']
            total_metadata['completion_tokens'] += answer_metadata['completion_tokens']
            answer_generation_end_time = time.perf_counter()
            answer_generation_total_time = answer_generation_end_time - answer_generation_start_time
            subquery_generation_total_time = None
            
            answer_response = '\n\n<answer>' + answer_response
            response = think_response + answer_response
            
            new_thought = get_thoughts(response)
            answer = get_answer(response)
            answer = "" if answer is None else answer
            new_subquery = None
            force_answer_flag = True

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
            max_gen_tokens=config.model.max_gen_tokens, 
            use_vllm=config.vllm.use_vllm,
            config=config
        )

        return model
