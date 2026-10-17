from typing import Tuple, List
import torch.cuda
import json
from llm_tools.safe_loader import LLMSafeLoader
from vllm import SamplingParams
from typing import Dict, Any, List, Tuple, Literal, Union, Optional, TypedDict
import logging
from packaging import version
import vllm
from vllm import TokensPrompt

logger = logging.getLogger(__name__)

class VLLMOffline:

    def __init__(self, model_name, max_model_len=4096, **kwargs):
        pipeline_parallel_size = 1
        tensor_parallel_size = kwargs.get('num_gpus', torch.cuda.device_count())
        if 'bnb' in model_name:
            kwargs['quantization'] = 'bitsandbytes'
            kwargs['load_format'] = 'bitsandbytes'
            tensor_parallel_size = 1
            pipeline_parallel_size = kwargs.get('num_gpus', torch.cuda.device_count())

        import os
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        self.model_name = model_name

        self.client = LLMSafeLoader.from_pretrained(
                          model=model_name,
                          tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
                          seed=kwargs.get('seed', 42), enable_prefix_caching=True,
                          enforce_eager=False, gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9), # modify: set gpu_memory_utilization to 0.95
                          max_model_len=max_model_len, quantization=kwargs.get('quantization', None), load_format=kwargs.get('load_format', 'auto'), trust_remote_code=True, 
                          max_num_seqs=kwargs.get('max_num_seqs', 100), 
                          dtype=kwargs.get('dtype', 'auto'), 
                          speculative_config=kwargs.get('speculative_config', None),
                          disable_log_stats=kwargs.get('disable_log_stats', True)
                          )

        self.tokenizer = self.client.get_tokenizer()
        
    def get_tokenizer(self):
        return self.tokenizer

    def reset_prefix_cache(self):
        self.client.reset_prefix_cache()

    def batch_infer_with_prompt_texts(self, prompt_texts: List[str], max_tokens=2048, 
                                      temperature=0, top_p=1.0, presence_penalty=0,
                                      json_schema=None, num_json_retry=0, json_retry_max_tokens=None,
                                      json_retry_repetition_penalty=1.0, choice=None, 
                                      stopping_sequences=None, skip_special_tokens=False):
        logger.info(f"Calling VLLM offline, # of sequences {len(prompt_texts)}")
        
        # structured outputs
        args = {}
        if version.parse(vllm.__version__) >= version.parse("0.11.0"):
            structuredOutputsParamsCls = vllm.sampling_params.StructuredOutputsParams
            param_key = 'structured_outputs'
        else:
            structuredOutputsParamsCls = vllm.sampling_params.GuidedDecodingParams
            param_key = 'guided_decoding'
        
        assert json_schema is None or choice is None, "`json_schema` and `choice` should not be set at the same time."
        if json_schema is not None:
            args[param_key] = structuredOutputsParamsCls(json=json_schema)
        elif choice is not None: 
            args[param_key] = structuredOutputsParamsCls(choice=choice)
        
        vllm_output = self.client.generate(prompts=prompt_texts,
                                           sampling_params=SamplingParams(
                                               max_tokens=max_tokens, 
                                               temperature=temperature,
                                               top_p=top_p,
                                               presence_penalty=presence_penalty,
                                               seed=42, # temperature is set to 0.0, so using greedy decoding, can ignore the seed
                                               stop=stopping_sequences, 
                                               include_stop_str_in_output=True, 
                                               skip_special_tokens=skip_special_tokens,
                                               **args
                                               ))

        all_responses = [completion.outputs[0].text for completion in vllm_output]
        all_prompt_tokens = [len(completion.prompt_token_ids) for completion in vllm_output]
        all_completion_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]
        all_total_tokens = [len(completion.prompt_token_ids) + len(completion.outputs[0].token_ids) 
                            for completion in vllm_output]
        logger.info(f"VLLM offline, # of sequences {len(prompt_texts)}, # of input tokens {sum(all_prompt_tokens)}, # of output tokens {sum(all_completion_tokens)}, max length {max(all_total_tokens)}")

        metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(all_prompt_tokens), 
            "max_length": max(all_total_tokens)
        }
        
        if json_schema is not None and num_json_retry > 0:
            # expand max_tokens for failure cases
            json_retry_max_tokens = json_retry_max_tokens if json_retry_max_tokens is not None else max_tokens
            current_indexes = list(range(len(all_responses)))
            current_responses = all_responses
            for i in range(num_json_retry):
                retry_prompt_texts = dict()
                failure_responses = []
                num_json_parse_failures = 0
                num_total = len(current_responses)
                for idx, response in zip(current_indexes, current_responses):
                    try:
                        _ = json.loads(response)
                    except Exception as e:
                        retry_prompt_texts[idx] = prompt_texts[idx]
                        failure_responses.append(response)
                        num_json_parse_failures += 1
                        
                if num_json_parse_failures == 0:
                    logger.info(f"All {num_total} responses parsed successfully.")
                    break

                logger.warning(f"Failed to parse {num_json_parse_failures} responses out of {num_total}. Retrying...")
                current_indexes = list(retry_prompt_texts.keys())
                current_prompt_texts = list(retry_prompt_texts.values())
                vllm_output = self.client.generate(prompts=current_prompt_texts,
                                                   sampling_params=SamplingParams(
                                                       max_tokens=json_retry_max_tokens, 
                                                       temperature=0, 
                                                       # add repetition panelty for failure cases
                                                       repetition_penalty=json_retry_repetition_penalty,
                                                       seed=42,  # temperature is set to 0.0, so using greedy decoding, can ignore the seed
                                                       stop=stopping_sequences, 
                                                       include_stop_str_in_output=True, 
                                                       skip_special_tokens=skip_special_tokens,
                                                       **args
                                                    ))
                current_responses = [completion.outputs[0].text for completion in vllm_output]
                current_prompt_tokens = [len(completion.prompt_token_ids) for completion in vllm_output]
                current_completion_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]
                current_total_tokens = [len(completion.prompt_token_ids) + len(completion.outputs[0].token_ids) 
                                        for completion in vllm_output]
                logger.info(f"VLLM offline retry {i+1}, # of messages {len(current_indexes)}, # of input tokens {sum(current_prompt_tokens)}, # of output tokens {sum(current_completion_tokens)}, max length {max(current_total_tokens)}")

                metadata["prompt_tokens"] += sum(current_prompt_tokens)
                metadata["completion_tokens"] += sum(current_completion_tokens)
                metadata["num_request"] += len(current_indexes)
                metadata["max_length"] = max(metadata["max_length"], max(current_total_tokens))

                for idx, response in zip(current_indexes, current_responses):
                    all_responses[idx] = response
                    
        torch.cuda.empty_cache()

        return all_responses, metadata

    def batch_infer_with_prompt_token_ids(self, prompt_token_ids, max_tokens=2048, 
                                          temperature=0, top_p=1.0, presence_penalty=0,
                                          json_schema=None, num_json_retry=0, json_retry_max_tokens=None, 
                                          json_retry_repetition_penalty=1.0, choice=None, 
                                          stopping_sequences=None, skip_special_tokens=False):
        logger.info(f"Calling VLLM offline, # of sequences {len(prompt_token_ids)}")

        # structured outputs
        args = {}
        if version.parse(vllm.__version__) >= version.parse("0.11.0"):
            structuredOutputsParamsCls = vllm.sampling_params.StructuredOutputsParams
            param_key = 'structured_outputs'
        else:
            structuredOutputsParamsCls = vllm.sampling_params.GuidedDecodingParams
            param_key = 'guided_decoding'
        
        assert json_schema is None or choice is None, "`json_schema` and `choice` should not be set at the same time."
        if json_schema is not None:
            args[param_key] = structuredOutputsParamsCls(json=json_schema)
        elif choice is not None: 
            args[param_key] = structuredOutputsParamsCls(choice=choice)

        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
        vllm_output = self.client.generate(prompts,
                                           sampling_params=SamplingParams(
                                               max_tokens=max_tokens, 
                                               temperature=temperature, 
                                               top_p=top_p,
                                               presence_penalty=presence_penalty,
                                               seed=42, # temperature is set to 0, so using greedy decoding, can ignore the seed
                                               # logprobs=5  # for debug
                                               stop=stopping_sequences, 
                                               include_stop_str_in_output=True, 
                                               skip_special_tokens=skip_special_tokens,
                                               **args
                                               ))

        all_responses = [completion.outputs[0].text for completion in vllm_output]
        all_prompt_tokens = [len(completion.prompt_token_ids) for completion in vllm_output]
        all_completion_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]
        all_total_tokens = [len(completion.prompt_token_ids) + len(completion.outputs[0].token_ids) 
                            for completion in vllm_output]
        logger.info(f"VLLM offline, # of sequences {len(prompt_token_ids)}, # of input tokens {sum(all_prompt_tokens)}, # of output tokens {sum(all_completion_tokens)}, max length {max(all_total_tokens)}")

        metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(all_prompt_tokens), 
            "max_length": max(all_total_tokens)
        }

        # modify: retry if JSON parsing fails
        if json_schema is not None and num_json_retry > 0:
            # expand max_tokens for failure cases
            json_retry_max_tokens = json_retry_max_tokens if json_retry_max_tokens is not None else max_tokens
            current_indexes = list(range(len(all_responses)))
            current_responses = all_responses
            for i in range(num_json_retry):
                retry_prompt_id_dict = dict()
                failure_responses = []
                num_json_parse_failures = 0
                num_total = len(current_responses)
                for idx, response in zip(current_indexes, current_responses):
                    try:
                        _ = json.loads(response)
                    except Exception as e:
                        retry_prompt_id_dict[idx] = prompt_token_ids[idx]
                        failure_responses.append(response)
                        num_json_parse_failures += 1
                        
                if num_json_parse_failures == 0:
                    logger.info(f"All {num_total} responses parsed successfully.")
                    break

                logger.warning(f"Failed to parse {num_json_parse_failures} responses out of {num_total}. Retrying...")
                current_indexes = list(retry_prompt_id_dict.keys())
                current_prompt_ids = list(retry_prompt_id_dict.values())
                prompts = [TokensPrompt(prompt_token_ids=ids) for ids in current_prompt_ids]
                vllm_output = self.client.generate(prompts,
                                                    sampling_params=SamplingParams(
                                                        max_tokens=json_retry_max_tokens, 
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        presence_penalty=presence_penalty,
                                                        # add repetition panelty for failure cases
                                                        repetition_penalty=json_retry_repetition_penalty,
                                                        seed=42,  # temperature is set to 0, so using greedy decoding, can ignore the seed
                                                        stop=stopping_sequences, 
                                                        include_stop_str_in_output=True, 
                                                        skip_special_tokens=skip_special_tokens,
                                                        **args
                                                    ))
                current_responses = [completion.outputs[0].text for completion in vllm_output]
                current_prompt_tokens = [len(completion.prompt_token_ids) for completion in vllm_output]
                current_completion_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]
                current_total_tokens = [len(completion.prompt_token_ids) + len(completion.outputs[0].token_ids) 
                                        for completion in vllm_output]
                logger.info(f"VLLM offline retry {i+1}, # of messages {len(current_indexes)}, # of input tokens {sum(current_prompt_tokens)}, # of output tokens {sum(current_completion_tokens)}, max length {max(current_total_tokens)}")

                metadata["prompt_tokens"] += sum(current_prompt_tokens)
                metadata["completion_tokens"] += sum(current_completion_tokens)
                metadata["num_request"] += len(current_indexes)
                metadata["max_length"] = max(metadata["max_length"], max(current_total_tokens))
                
                for idx, response in zip(current_indexes, current_responses):
                    all_responses[idx] = response

        torch.cuda.empty_cache()

        return all_responses, metadata

    def infer_with_prompt_texts(self, prompt_texts: str, max_tokens=2048, 
                                temperature=0, top_p=1.0, presence_penalty=0,
                                 json_schema=None, num_json_retry=0, json_retry_max_tokens=None,
                                 json_retry_repetition_penalty=1.0, choice=None, skip_special_tokens=False):
        prompt_texts = [prompt_texts]
        
        return self.batch_infer_with_prompt_texts(prompt_texts,
                                                  max_tokens=max_tokens,
                                                  temperature=temperature,
                                                  top_p=top_p,
                                                  presence_penalty=presence_penalty,
                                                  json_schema=json_schema,
                                                  num_json_retry=num_json_retry,
                                                  json_retry_max_tokens=json_retry_max_tokens,
                                                  json_retry_repetition_penalty=json_retry_repetition_penalty,
                                                  choice=choice,
                                                  skip_special_tokens=skip_special_tokens)

    def infer_with_prompt_token_ids(self, prompt_token_ids, max_tokens=2048, 
                                    temperature=0, top_p=1.0, presence_penalty=0,
                                    json_schema=None, num_json_retry=0, json_retry_max_tokens=None, 
                                    json_retry_repetition_penalty=1.0, choice=None, skip_special_tokens=False):
        prompt_token_ids = [prompt_token_ids]
        
        return self.batch_infer_with_prompt_token_ids(prompt_token_ids, 
                                                      max_tokens=max_tokens, 
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      presence_penalty=presence_penalty,
                                                      json_schema=json_schema,
                                                      num_json_retry=num_json_retry, 
                                                      json_retry_max_tokens=json_retry_max_tokens, 
                                                      json_retry_repetition_penalty=json_retry_repetition_penalty, 
                                                      choice=choice,
                                                      skip_special_tokens=skip_special_tokens)

    def get_metrics(self):
        try:
            metrics = self.client.get_metrics()
        except Exception as e:
            logger.warning(f"Failed to get metrics from vLLM client: {e}")
            return None
        
        return metrics

    