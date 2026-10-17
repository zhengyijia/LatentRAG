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


def load_searchr1_tokenizer(config): 
    tokenizer = AutoTokenizerSafeLoader.from_pretrained(
        "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token is None:
        tokenizer.bos_token = ''
    tokenizer.padding_side = 'left'

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

# for training draft model for SearchR1
class SearchR1DraftSFTModel(nn.Module):
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

        del outputs
        del logits
        # torch.cuda.empty_cache()

        gen_scaling_factor = 1.0
        gen_scaling_factor_without_spec_tokens = 1.0
        if self.is_ddp:
            # gather num_elements for scaling results
            avg_gen_num_elements = self._dist_gather_scalar_mean(gen_num_elements)

            gen_scaling_factor = gen_num_elements / avg_gen_num_elements if avg_gen_num_elements > 0 else 0
            
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
        )

    def _dist_gather_scalar_mean(self, value: torch.Tensor):
        value_sum = value.clone()
        dist.all_reduce(value_sum, op=dist.ReduceOp.SUM)
        return value_sum.item() / self.world_size


    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable()
        # if hasattr(self, 'encoder') and self.encoder is not None:
        #     self.encoder.gradient_checkpointing_enable(**kwargs)

    @classmethod
    def get_special_added_tokens(cls, config): 
        spec_tokens = dict()

        # begin and end of decode thought/subquery
        spec_tokens['bot_token'] = '<think>'
        spec_tokens['eot_token'] = '</think>'
        spec_tokens['bosq_token'] = '\n\n<search> '
        spec_tokens['eosq_token'] = ' </search>\n\n'

        # tokens for retrieved docs
        spec_tokens['bod_token'] = '<information>'
        spec_tokens['eod_token'] = '</information>'
        
        # tokens for final answer
        spec_tokens['boa_token'] = '\n\n<answer> '
        spec_tokens['eoa_token'] = ' </answer>'

        # create a list of all special tokens
        spec_token_list = sorted({
            t for v in spec_tokens.values() for t in (v if isinstance(v, list) else [v])
        })

        return spec_tokens, spec_token_list
    
    @classmethod
    def get_special_prompts(cls, config): 
        spec_tokens, _ = cls.get_special_added_tokens(config)
        spec_prompts = dict()
        spec_prompts['query_prompt'] = prompt = ("Answer the given question. "
                            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
                            "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. "
                            "You can search as many times as your want. "
                            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {query}\n")

        return spec_prompts
        

    @classmethod
    def _resize_model_token_embeddings(cls, model, config): 
        model_config = AutoConfigSafeLoader.from_pretrained(
            "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        )
        vocab_size = model_config.vocab_size
        model.resize_token_embeddings(vocab_size)

    @classmethod
    def build(cls, config):
        tokenizer = load_searchr1_tokenizer(
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
            
        cls._resize_model_token_embeddings(base_model, config)
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
            tokenizer = load_searchr1_tokenizer(
                config=config
            )
            base_model = AutoModelForCausalLMSafeLoader.from_pretrained(
                config.model.model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=config.model.attn_implementation, 
                device_map="auto"
            )
            cls._resize_model_token_embeddings(base_model, config)
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
