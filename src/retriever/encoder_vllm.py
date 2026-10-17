import os
import re
import numpy as np
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import torch
from torch import nn
import logging
from packaging import version
import vllm

from llm_tools.safe_loader import LLMSafeLoader, AutoTokenizerSafeLoader

logger = logging.getLogger(__name__)

class Encoder_vllm(nn.Module):
    """
    Encoder class for encoding queries using a specified model. NOT NORMALIZED!

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path to the model.
        max_length (int): The maximum length of the input sequences.

    Methods:
        encode(query_list: List[str], is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
    """

    def __init__(self, model_name, base_model, tokenizer, max_length, 
                 return_dtype=torch.float32,
                 silent=False):
        super().__init__()
        self.model_name = model_name
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.silent = silent
        self.return_dtype = return_dtype
        
    def torch_to_numpy_dtype(self, torch_dtype):
        mapping = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.bfloat16: np.float32,  # Use higher precision for bfloat16
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.bool: np.bool_
        }
        return mapping[torch_dtype]

    def format_inputs(self, query_list: Union[List[str], str], is_query=True):
        if isinstance(query_list, str):
            query_list = [query_list]
        
        if "qwen3-embedding" in self.model_name.lower():
            if is_query:
                query_list = [f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}' for query in query_list]
        elif "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]
                
        return query_list

    def gen_embeddings(self, inputs, return_tensors=False):
        outputs = self.base_model.embed(inputs, truncate_prompt_tokens=self.max_length)
        if not return_tensors:
            return_dtype = self.torch_to_numpy_dtype(self.return_dtype)
            query_emb = np.array([o.outputs.embedding for o in outputs], dtype=return_dtype)
        else:
            query_emb = torch.tensor([o.outputs.embedding for o in outputs], dtype=self.return_dtype)

        return query_emb

    def single_batch_encode(self, query_list: Union[List[str], str], is_query=True, return_tensors=False) -> np.ndarray:
        inputs = self.format_inputs(query_list, is_query)
        query_emb = self.gen_embeddings(inputs, return_tensors=return_tensors)

        return query_emb
    
    @classmethod
    def load(cls, config, silent=True, **kwargs):
        torch_dtype = torch.float32
        if getattr(config.training, 'bf16', False): 
            torch_dtype = torch.bfloat16
        elif getattr(config.training, 'fp16', False):
            torch_dtype = torch.float16
        
        pipeline_parallel_size = 1
        tensor_parallel_size = kwargs.get('num_gpus', torch.cuda.device_count())
        
        args = {}
        if version.parse(vllm.__version__) >= version.parse("0.11.1"):
            args['convert'] = 'embed'
        else:
            args['task'] = 'embed'
        
        base_model = LLMSafeLoader.from_pretrained(
            model=config.retriever.encoder_name_or_path,
            tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
            seed=kwargs.get('seed', 42), 
            gpu_memory_utilization=config.vllm.encoder_model_gpu_memory_utilization,
            max_model_len=config.vllm.encoder_model_max_model_len,
            max_num_seqs=config.vllm.encoder_model_max_num_seqs,
            dtype=torch_dtype,
            enforce_eager=True, 
            **args)
        
        tokenizer = AutoTokenizerSafeLoader.from_pretrained(config.retriever.encoder_name_or_path, use_fast=True)
        
        encoder = cls(model_name=config.retriever.retriever_tag,
                      base_model=base_model,
                      tokenizer=tokenizer,
                      max_length=config.retriever.encoder_query_max_length,
                      return_dtype=torch_dtype,
                      silent=silent)
        
        return encoder
