from typing import Tuple, List
import torch.cuda
import json
from llm_tools.safe_loader import LLMSafeLoader
from vllm import SamplingParams
from typing import Dict, Any, List, Tuple, Literal, Union, Optional, TypedDict, AsyncGenerator
import logging
from packaging import version
import vllm
from vllm import TokensPrompt
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import uuid
import time

logger = logging.getLogger(__name__)

class VLLMOfflineAsync:
    def __init__(self, model_name, max_model_len=4096, **kwargs):
        pipeline_parallel_size = 1
        tensor_parallel_size = kwargs.get('num_gpus', torch.cuda.device_count())

        import os
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        self.model_name = model_name

        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
            seed=kwargs.get('seed', 42), enable_prefix_caching=True,
            enforce_eager=False, gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
            max_model_len=max_model_len, quantization=kwargs.get('quantization', None), load_format=kwargs.get('load_format', 'auto'), trust_remote_code=True, 
            max_num_seqs=kwargs.get('max_num_seqs', 100), 
            dtype=kwargs.get('dtype', 'auto')
        )

        self.client = AsyncLLMEngine.from_engine_args(
            engine_args
        )

        self.tokenizer = self.client.get_tokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def reset_prefix_cache(self):
        self.client.reset_prefix_cache()

    async def infer_one(self, prompt_text: str, max_tokens=2048, temperature=0, top_p=1.0, 
                        presence_penalty=0, stopping_sequences=None, skip_special_tokens=False):
    
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            seed=42,
            stop=stopping_sequences,
            include_stop_str_in_output=True,
            skip_special_tokens=skip_special_tokens,
        )
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        generator = self.client.generate(
            prompt=prompt_text,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        final_output = None
        async for output in generator:
            final_output = output

        end = time.perf_counter()

        response = final_output.outputs[0].text
        num_prompt_tokens = len(final_output.prompt_token_ids)
        num_completion_tokens = len(final_output.outputs[0].token_ids)

        return {
            "response": response,
            "latency": end - start,
            "prompt_tokens": num_prompt_tokens,
            "completion_tokens": num_completion_tokens,
            "total_tokens": num_prompt_tokens + num_completion_tokens,
        }

    async def _batch_infer_async(
        self,
        prompt_texts: List[str],
        max_tokens=2048,
        temperature=0,
        top_p=1.0,
        presence_penalty=0,
        stopping_sequences=None,
        skip_special_tokens=False,
    ):
        logger.info(
            f"submit {len(prompt_texts)} requests"
        )

        async def run_one(idx, prompt):
            result = await self.infer_one(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                stopping_sequences=stopping_sequences,
                skip_special_tokens=skip_special_tokens,
            )

            result["index"] = idx

            return result

        tasks = [
            asyncio.create_task(
                run_one(i, p)
            )
            for i, p in enumerate(prompt_texts)
        ]


        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)

        return results


    def batch_infer_with_prompt_texts(
        self,
        prompt_texts: List[str],
        **kwargs
    ):

        try:
            loop = asyncio.get_event_loop()

        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)


        return loop.run_until_complete(
            self._batch_infer_async(
                prompt_texts,
                **kwargs
            )
        )
