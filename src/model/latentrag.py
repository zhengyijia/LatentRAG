import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List, Tuple, Any
import numpy as np
import time
import random
from omegaconf import DictConfig
from contextlib import nullcontext

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import ModelOutput
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from safetensors.torch import save_file

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


# ================== Projector Modules ==================

class ProjectorBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(output_dim)

        # self.use_residual = (input_dim == output_dim)

    def forward(self, x):
        out = self.fc(x)
        out = self.act(out)
        out = self.norm(out)

        # if self.use_residual:
        #     out = out + x

        return out

class SelfAttnAdapter(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=4
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.norm1 = nn.LayerNorm(input_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, self.output_dim)
        )

    def forward(self, hiddens):
        # Self-attention block
        x = self.norm1(hiddens)
        attn_out, _ = self.attn(x, x, x)
        x = hiddens + attn_out

        # FFN block
        y = self.norm2(x)
        y = self.ffn(y)

        return y
    
class CrossAttnAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, num_output_tokens=16, num_heads=4):
        super().__init__()

        self.num_output_tokens = num_output_tokens

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_output_tokens, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def forward(self, hiddens):
        B = hiddens.size(0)

        queries = self.query_tokens.expand(B, -1, -1)
        
        hiddens = self.norm1(hiddens)

        attn_out, _ = self.attn(
            queries,   # Q
            hiddens,   # K
            hiddens    # V
        )

        y = self.norm2(attn_out)
        y = self.ffn(y)

        return y
    
def build_projector(input_dim, output_dim, projector_type='linear'):
    if projector_type == 'linear':
        model = nn.Linear(input_dim, output_dim)
        model.weight.data.fill_(0.0)
        return model

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        
        dim = input_dim
        hidden_dim = 2*input_dim
        modules = []
        for _ in range(mlp_depth - 1):
            block = ProjectorBlock(dim, hidden_dim)
            modules.append(block)
            dim = hidden_dim
        modules.append(nn.Linear(dim, output_dim))
        modules[-1].weight.data.fill_(0.0)
        return nn.Sequential(*modules)
    
    if projector_type == 'self_attn':
        model = SelfAttnAdapter(input_dim, output_dim)
        return model
    
    if projector_type == 'cross_attn':
        model = CrossAttnAdapter(input_dim, output_dim)
        return model

    raise ValueError(f'Unknown projector type: {projector_type}')

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

    _, spec_token_list = LatentRAG.get_special_added_tokens(config)
    tokenizer.add_special_tokens({"additional_special_tokens": spec_token_list})

    return tokenizer

def compute_segment_lm_metrics(logits, labels, input_ids, label_smoothing=0.0,
                               start_token_ids=None, end_token_ids=None, spec_token_ids=None,
                               ignore_index=-100):
    B, T = input_ids.shape
    device = input_ids.device
    
    range_tensor = torch.arange(T, device=device).unsqueeze(0)
    segment_mask = torch.ones_like(input_ids, dtype=torch.bool)
    
    if start_token_ids is not None:
        start_token_ids = torch.tensor(start_token_ids, device=input_ids.device)
        match = (input_ids.unsqueeze(-1) == start_token_ids).any(dim=-1)
        start_pos = match.float().argmax(dim=1)
        segment_mask &= range_tensor >= start_pos.unsqueeze(1)
        
    if end_token_ids is not None:
        end_token_ids = torch.tensor(end_token_ids, device=input_ids.device)
        match = (input_ids.unsqueeze(-1) == end_token_ids).any(dim=-1)
        end_pos = match.float().argmax(dim=1)
        segment_mask &= range_tensor <= end_pos.unsqueeze(1)
    
    # if start_token_id is not None:
    #     start_pos = (input_ids == start_token_id).float().argmax(dim=1)
    #     segment_mask &= range_tensor >= start_pos.unsqueeze(1)
        
    # if end_token_id is not None:
    #     end_pos = (input_ids == end_token_id).float().argmax(dim=1)
    #     segment_mask &= range_tensor <= end_pos.unsqueeze(1)

    masked_labels = labels.clone()
    masked_labels[~segment_mask] = ignore_index
    
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = masked_labels[..., 1:].contiguous().view(-1)

    valid_mask = shift_labels != ignore_index
    shift_logits = shift_logits[valid_mask]
    shift_labels = shift_labels[valid_mask]
    
    num_elements = shift_logits.shape[0]
    loss = nn.functional.cross_entropy(shift_logits, shift_labels, 
                                       label_smoothing=label_smoothing)
    
    preds = shift_logits.argmax(dim=-1)
    accuracy = (preds == shift_labels).float().mean()

    accuracy_without_spec_tokens = None
    num_elements_without_spec_tokens = num_elements
    if spec_token_ids is not None:
        spec_mask = ~torch.isin(shift_labels, spec_token_ids)
        num_elements_without_spec_tokens = spec_mask.float().sum().item()
        accuracy_without_spec_tokens = (
            ((preds == shift_labels) & spec_mask).float().sum()
            / num_elements_without_spec_tokens
        )
    else:
        accuracy_without_spec_tokens = accuracy
        
    return {
        "num_elements": torch.tensor(num_elements, device=device, dtype=torch.long),
        "num_elements_without_spec_tokens": torch.tensor(num_elements_without_spec_tokens, device=device, dtype=torch.long),
        "loss": loss,
        "accuracy": accuracy,
        "accuracy_without_spec_tokens": accuracy_without_spec_tokens,
    }

def sample_self_decode_data(inputs_embeds, input_batch, max_token_num=-1):
    n, m = inputs_embeds.shape[:2]
    if max_token_num == -1 or n * m <= max_token_num:
        return inputs_embeds, input_batch
    
    sample_n = max_token_num // m
    indices = torch.randperm(n, device=inputs_embeds.device)[:sample_n]

    inputs_embeds = inputs_embeds[indices]
    for key in input_batch:
        input_batch[key] = input_batch[key][indices]

    return inputs_embeds, input_batch

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

class LatentRAG(nn.Module):
    def __init__(self, 
                 base_model: PreTrainedModel,
                 llm_projector: nn.Module,
                 llm_self_thought_projector: nn.Module,
                 llm_self_subquery_projector: nn.Module,
                 tokenizer: PreTrainedTokenizerBase, 
                 config: DictConfig, 
                 ):
        super().__init__()
        self.config = base_model.config  # config is the hf config info of the base model
        self.conf = config  # conf is the config of LatentRAG
        self.base_model = base_model
        self.base_embedding_layer = base_model.get_input_embeddings()
        self.llm_projector = llm_projector
        self.llm_self_thought_projector = llm_self_thought_projector
        self.llm_self_subquery_projector = llm_self_subquery_projector
        
        self.title_in_first_line = getattr(self.conf.data, 'title_in_first_line', True)

        self.align_loss = self.conf.custom_training.align_loss
        # for nce loss
        self.pos_k = self.conf.retriever.top_k
        self.neg_k = self.conf.custom_training.neg_k
        random_neg_k = self.conf.custom_training.get('random_neg_k', 0)
        self.neg_k += random_neg_k
        self.temperature = self.conf.custom_training.temperature
        # for cosine_nce_loss
        self.align_mse_loss_weight = getattr(self.conf.custom_training, 'align_mse_loss_weight', 1.0)
        self.align_nce_loss_weight = getattr(self.conf.custom_training, 'align_nce_loss_weight', 1.0)

        self.align_loss_weight = getattr(self.conf.custom_training, 'align_loss_weight', 1.0)
        self.gen_loss_weight = getattr(self.conf.custom_training, 'gen_loss_weight', 1.0)
        self.self_decode_thought_gen_loss_weight = getattr(self.conf.custom_training, 'self_decode_thought_gen_loss_weight', 1.0)
        self.self_decode_subquery_gen_loss_weight = getattr(self.conf.custom_training, 'self_decode_subquery_gen_loss_weight', 1.0)

        self.align_retriever = getattr(self.conf.custom_training, 'align_retriever', True)
        self.decode_thought = getattr(self.conf.custom_training, 'decode_thought', False)
        self.decode_subquery = getattr(self.conf.custom_training, 'decode_subquery', False)
        
        self.llm_only_query_embedding = getattr(self.conf.model, 'llm_only_query_embedding', False) # use LLM hidden states as retriever embedding (for ablation)
        
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

        # only used to limit the total token numbers for thought and subquery decoding during training
        self.max_seq_length = config.data.get('max_seq_length', -1)

        self.use_chat_template = config.model.use_chat_template
        
        self.base_model_trainable = config.model.base_model_trainable

        self.update_ddp_state()
    
    def update_ddp_state(self):
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
    
    # for generating supervised signal
    def set_encoder(self, encoder): 
        self.encoder = encoder
        
    def get_encoder(self): 
        return self.encoder
    
    def set_reference_encoder(self, reference_encoder): 
        self.reference_encoder = reference_encoder
        
    def get_reference_encoder(self): 
        return self.reference_encoder
            
    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, 
                llm_input_batches: Dict[str, Any] = None,
                retrieval_data: Dict[str, Any] = None) -> TrainingOutput:
        
        if not self.reference_encoder.trainable: 
            self.reference_encoder.eval() # ensure the target is accurate
            
        if not self.encoder.trainable:
            self.encoder.eval()  # stabilize the training process
        
        input_batch = llm_input_batches['input_batch']
        self_decode_thought_input_batch = llm_input_batches['self_decode_thought_input_batch']
        self_decode_subquery_input_batch = llm_input_batches['self_decode_subquery_input_batch']
        
        has_subquery = len([item for sublist in retrieval_data['subqueries'] for item in sublist]) > 0
        has_subquery = torch.tensor(has_subquery, device=self.base_model.device, dtype=torch.bool)
        if not has_subquery:
            # add dummy data to avoid code branches
            # make sure all forward code can run under deepspeed zero3
            self_decode_subquery_input_batch = {
                'input_ids': torch.ones((1, 2), dtype=torch.long, device=self.base_model.device),
                'attention_mask': torch.ones((1, 2), dtype=torch.long, device=self.base_model.device),
                'labels': torch.ones((1, 2), dtype=torch.long, device=self.base_model.device),
                'position_ids': torch.arange(2, dtype=torch.long, device=self.base_model.device).unsqueeze(0)
            }
            retrieval_data = {
                'query_ids': ['dummy_id'], 
                'queries': ['dummy query'],
                'thoughts': [['dummy thought']],
                'subqueries': [['dummy subquery']],
                'pos_chunk_items': [[[{'chunk_id': f'dummy_pos_chunk_id_{i}', 'text': f'dummy pos chunk text {i}', 'score': 0.8} for i in range(self.pos_k)]]],
                'hard_neg_chunk_items': [[[{'chunk_id': f'dummy_neg_chunk_id_{i}', 'text': f'dummy neg chunk text {i}', 'score': 0.2} for i in range(self.neg_k)]]]
            } 
        
        # forward LLM
        inputs_embeds = self.base_embedding_layer(input_batch['input_ids'])

        if torch.is_grad_enabled():
            context = torch.no_grad() if not self.base_model_trainable else torch.enable_grad()
        else:
            context = nullcontext()
        with context:
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=input_batch['attention_mask'],
                position_ids=input_batch['position_ids'],
                output_hidden_states=True,
                use_cache=False
            )
        logits, last_hidden_states = outputs.logits, outputs.hidden_states[-1]
        # hidden_states = outputs.hidden_states

        # # use float32 for better stability
        # logits = logits.float()

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

        # collect latent thought llm features
        latent_indices = (
            (input_batch['input_ids'] == self.spec_token_ids['latent_end_token'])
        ).nonzero()
        num_latent_tokens = len(self.spec_token_ids['latent_tokens'])
        offsets = torch.arange(num_latent_tokens - 1, -1, -1, device=latent_indices.device)  # [0, ..., num_latent_tokens-1]
        token_indices = latent_indices[:, 1].unsqueeze(1) - offsets.unsqueeze(0)  # [N, num_latent_tokens]
        batch_indices = latent_indices[:, 0].unsqueeze(1).expand(-1, num_latent_tokens)
        latent_llm_features = last_hidden_states[batch_indices, token_indices, :]
        
        # collect latent subquery llm features
        latent_subquery_indices = (
            (input_batch['input_ids'] == self.spec_token_ids['latent_subquery_end_token'])
        ).nonzero()
        num_latent_subquery_tokens = len(self.spec_token_ids['latent_subquery_tokens'])
        if latent_subquery_indices.size(0) == 0:
            latent_subquery_llm_features = torch.zeros((1, num_latent_subquery_tokens, last_hidden_states.size(-1)), 
                                                       device=last_hidden_states.device, dtype=last_hidden_states.dtype)
        else:
            offsets = torch.arange(num_latent_subquery_tokens - 1, -1, -1, device=latent_subquery_indices.device)  # [0, ..., num_latent_subquery_tokens-1]
            token_indices = latent_subquery_indices[:, 1].unsqueeze(1) - offsets.unsqueeze(0)  # [N, num_latent_subquery_tokens]
            batch_indices = latent_subquery_indices[:, 0].unsqueeze(1).expand(-1, num_latent_subquery_tokens)
            latent_subquery_llm_features = last_hidden_states[batch_indices, token_indices, :]

        del outputs
        del logits
        del last_hidden_states
        # torch.cuda.empty_cache()

        # self decode thought
        self_decode_thought_metrics = None
        if self.decode_thought and self_decode_thought_input_batch is not None:
            # add self decode loss
            latent_llm_self_decode_thought_embeddings = self.llm_self_thought_projector(latent_llm_features)
            # construct inputs_embeds
            all_self_decode_thought_inputs_embeds = self.base_embedding_layer(self_decode_thought_input_batch['input_ids'])
            self_decode_thought_latent_begin_indices = (
                (self_decode_thought_input_batch['input_ids'] == self.spec_token_ids['latent_begin_token'])
            ).nonzero()
            replace_positions = self_decode_thought_latent_begin_indices[:, 1].unsqueeze(1) + \
                torch.arange(num_latent_tokens, device=self_decode_thought_latent_begin_indices.device).unsqueeze(0)
            all_self_decode_thought_inputs_embeds[self_decode_thought_latent_begin_indices[:, 0].unsqueeze(1), replace_positions] = latent_llm_self_decode_thought_embeddings
            # sampling decode data to avoid OOM
            all_self_decode_thought_inputs_embeds, self_decode_thought_input_batch = sample_self_decode_data(
                all_self_decode_thought_inputs_embeds, 
                self_decode_thought_input_batch, 
                max_token_num=self.max_seq_length)
            # forward
            with context:
                self_decode_thought_outputs = self.base_model(
                    inputs_embeds=all_self_decode_thought_inputs_embeds,
                    attention_mask=self_decode_thought_input_batch['attention_mask'],
                    output_hidden_states=False,
                    use_cache=False
                )
            self_decode_thought_logits = self_decode_thought_outputs.logits
            
            # compute metrics
            self_decode_thought_metrics = compute_segment_lm_metrics(
                self_decode_thought_logits, self_decode_thought_input_batch['labels'], self_decode_thought_input_batch['input_ids'],
                label_smoothing=self.conf.custom_training.get('label_smoothing', 0.0),
                start_token_ids=[self.spec_token_ids['bot_token']],
                end_token_ids=None,
                spec_token_ids=self.spec_token_id_tensor
            )
            
            del self_decode_thought_outputs
            del self_decode_thought_logits
            
        # self decode subquery
        self_decode_subquery_metrics = None
        if self.decode_subquery and self_decode_subquery_input_batch is not None:
            # add self decode loss
            latent_llm_self_decode_subquery_embeddings = self.llm_self_subquery_projector(latent_subquery_llm_features)
            # construct inputs_embeds
            all_self_decode_subquery_inputs_embeds = self.base_embedding_layer(self_decode_subquery_input_batch['input_ids'])
            self_decode_subquery_latent_begin_indices = (
                (self_decode_subquery_input_batch['input_ids'] == self.spec_token_ids['latent_subquery_begin_token'])
            ).nonzero()
            replace_positions = self_decode_subquery_latent_begin_indices[:, 1].unsqueeze(1) + \
                torch.arange(num_latent_subquery_tokens, device=self_decode_subquery_latent_begin_indices.device).unsqueeze(0)
            all_self_decode_subquery_inputs_embeds[self_decode_subquery_latent_begin_indices[:, 0].unsqueeze(1), replace_positions] = latent_llm_self_decode_subquery_embeddings
            # sampling decode data to avoid OOM
            all_self_decode_subquery_inputs_embeds, self_decode_subquery_input_batch = sample_self_decode_data(
                all_self_decode_subquery_inputs_embeds, 
                self_decode_subquery_input_batch, 
                max_token_num=self.max_seq_length)
            # forward
            with context:
                self_decode_subquery_outputs = self.base_model(
                    inputs_embeds=all_self_decode_subquery_inputs_embeds,
                    attention_mask=self_decode_subquery_input_batch['attention_mask'],
                    output_hidden_states=False,
                    use_cache=False
                )
            self_decode_subquery_logits = self_decode_subquery_outputs.logits
            
            # compute metrics
            self_decode_subquery_metrics = compute_segment_lm_metrics(
                self_decode_subquery_logits, self_decode_subquery_input_batch['labels'], self_decode_subquery_input_batch['input_ids'],
                label_smoothing=self.conf.custom_training.get('label_smoothing', 0.0),
                start_token_ids=[self.spec_token_ids['bosq_token']],
                end_token_ids=None,
                spec_token_ids=self.spec_token_id_tensor
            )
            
            del self_decode_subquery_outputs
            del self_decode_subquery_logits

        if self.align_retriever:
            # subquery embeddings
            flat_subqueries = [item for sublist in retrieval_data['subqueries'] for item in sublist]
            # pos neg chunks
            # add postive chunks
            flat_pos_neg_chunk_ids = [chunk_item['chunk_id'] for sublist in retrieval_data['pos_chunk_items'] 
                                for chunk_items in sublist for chunk_item in chunk_items]
            flat_pos_neg_chunk_texts = [chunk_item['text'] for sublist in retrieval_data['pos_chunk_items'] 
                                    for chunk_items in sublist for chunk_item in chunk_items]
            # add negative chunks
            flat_pos_neg_chunk_ids.extend([chunk_item['chunk_id'] for sublist in retrieval_data['hard_neg_chunk_items']
                                            for chunk_items in sublist for chunk_item in chunk_items])
            flat_pos_neg_chunk_texts.extend([chunk_item['text'] for sublist in retrieval_data['hard_neg_chunk_items']
                                            for chunk_items in sublist for chunk_item in chunk_items])

            # subquery embeddings
            flat_subquery_embeddings = self.reference_encoder.single_batch_encode(flat_subqueries, is_query=True, return_tensors=True)
            flat_subquery_embeddings = flat_subquery_embeddings.float() # use float32 for better stability
            normalized_flat_subquery_embeddings = nn.functional.normalize(flat_subquery_embeddings, p=2, dim=-1)

            # positive and negative chunk embeddings
            flat_pos_neg_chunk_embeddings = self.reference_encoder.single_batch_encode(flat_pos_neg_chunk_texts, is_query=False, return_tensors=True)
            flat_pos_neg_chunk_embeddings = flat_pos_neg_chunk_embeddings.float() # use float32 for better stability
            flat_pos_neg_chunk_embeddings = nn.functional.normalize(flat_pos_neg_chunk_embeddings, p=2, dim=-1)

            if self.is_ddp:
                # gathered_flat_subquery_embeddings, subquery_offset = self._dist_gather_tensor_variable_batch(flat_subquery_embeddings, return_offset=True)
                gathered_flat_pos_neg_chunk_embeddings, pos_neg_chunk_offset = self._dist_gather_tensor_variable_batch(flat_pos_neg_chunk_embeddings, return_offset=True)
                gathered_flat_pos_neg_chunk_ids = self._dist_gather_list(flat_pos_neg_chunk_ids)
            else:
                gathered_flat_pos_neg_chunk_embeddings = flat_pos_neg_chunk_embeddings
                pos_neg_chunk_offset = 0
                gathered_flat_pos_neg_chunk_ids = flat_pos_neg_chunk_ids

            # gen embeddings
            latent_subquery_llm_embeddings = self.llm_projector(latent_subquery_llm_features)
            if self.llm_only_query_embedding: # for ablation
                gen_embeddings = latent_subquery_llm_embeddings[:, -1, :]
            else:
                gen_embeddings = self.encoder.gen_embeddings_with_latent_inputs(latent_subquery_llm_embeddings, is_query=True, return_tensors=True, no_grad=False)
            gen_embeddings = gen_embeddings.float() # use float32 for better stability
            normalized_gen_embeddings = nn.functional.normalize(gen_embeddings, p=2, dim=-1)

            align_num_elements = torch.tensor(gen_embeddings.shape[0], device=gen_embeddings.device, dtype=torch.long)
            # MSE loss
            # cos_sim = nn.functional.cosine_similarity(gen_embeddings, flat_subquery_embeddings, dim=-1)
            # align_cosine_loss = 1 - cos_sim.mean()
            align_mse_loss = ((normalized_gen_embeddings - normalized_flat_subquery_embeddings) ** 2).sum(dim=1).mean()
            align_unnorm_mse_loss = ((gen_embeddings - flat_subquery_embeddings) ** 2).sum(dim=1).mean()
            
            # For NCE loss
            q_reps = normalized_gen_embeddings
            p_reps = gathered_flat_pos_neg_chunk_embeddings
            scores = torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature
            n_q_reps = q_reps.shape[0]
            # calculate mask
            neg_mask = torch.ones_like(scores, dtype=torch.bool)
            topk_pos_indices = (
                pos_neg_chunk_offset + torch.arange(n_q_reps, device=q_reps.device).unsqueeze(1) * self.pos_k
                + torch.arange(self.pos_k, device=q_reps.device).unsqueeze(0)
            )
            neg_mask.scatter_(1, topk_pos_indices, False)
            # mask negative samples that are also positive samples
            p_chunk_ids = gathered_flat_pos_neg_chunk_ids
            id2idx = {chunk_id: idx for idx, chunk_id in enumerate(set(p_chunk_ids))}
            p_chunk_idxs = torch.tensor([id2idx[chunk_id] for chunk_id in p_chunk_ids], device=q_reps.device)
            topk_chunk_idxs = p_chunk_idxs[topk_pos_indices]
            false_neg_mask =  (topk_chunk_idxs.unsqueeze(2) == p_chunk_idxs[None, None, :]).any(dim=1)
            neg_mask = neg_mask & (~false_neg_mask)
            # (1) Top-1 InfoNCE loss
            top1_pos_scores = scores[torch.arange(scores.size(0)), topk_pos_indices[:, 0]]
            masked_neg_scores = scores.masked_fill(~neg_mask, float('-inf'))
            log_all = torch.logsumexp(torch.cat([top1_pos_scores.unsqueeze(1), masked_neg_scores], dim=1), dim=1)
            align_top1_nce_loss = -(top1_pos_scores - log_all).mean()
            # (2) Top-k InfoNCE loss
            ## Multi-positive InfoNCE
            topk_pos_scores = scores.gather(1, topk_pos_indices)
            log_pos = torch.logsumexp(topk_pos_scores, dim=1)
            log_all = torch.logsumexp(torch.cat([topk_pos_scores, masked_neg_scores], dim=1), dim=1)
            align_topk_nce_loss = -(log_pos - log_all).mean()
            # (3) Max-Positive InfoNCE loss
            max_pos_scores, _ = topk_pos_scores.max(dim=1)
            log_all = torch.logsumexp(torch.cat([max_pos_scores.unsqueeze(1), masked_neg_scores], dim=1), dim=1)
            align_max_pos_nce_loss = -(max_pos_scores - log_all).mean()
            # (4) Min-Positive InfoNCE loss
            min_pos_scores, _ = topk_pos_scores.min(dim=1)
            log_all = torch.logsumexp(torch.cat([min_pos_scores.unsqueeze(1), masked_neg_scores], dim=1), dim=1)
            align_min_pos_nce_loss = -(min_pos_scores - log_all).mean()
            # (5) Per Top-k InfoNCE loss
            pos_scores_flat = topk_pos_scores.reshape(-1)
            neg_scores_flat = masked_neg_scores.unsqueeze(1).expand(-1, 3, -1).reshape(pos_scores_flat.shape[0], -1)
            log_all = torch.logsumexp(
                torch.cat([pos_scores_flat.unsqueeze(1), neg_scores_flat], dim=1),
                dim=1
            )
            align_per_topk_nce_loss = -(pos_scores_flat - log_all).mean()
            # (6) KL loss - distribution matching
            # teacher scores
            with torch.no_grad():
                teacher_scores = torch.matmul(
                    normalized_flat_subquery_embeddings, p_reps.transpose(0, 1)
                ) / self.temperature
            student_log_probs = torch.log_softmax(scores, dim=1)
            with torch.no_grad():
                teacher_probs = torch.softmax(teacher_scores, dim=1)
            align_kl_rank_loss = torch.sum(
                teacher_probs * (torch.log(teacher_probs + 1e-9) - student_log_probs),
                dim=1
            ).mean()
            
            if self.align_loss == 'mse_loss':
                align_loss = align_mse_loss * self.align_mse_loss_weight
            elif self.align_loss == 'unnorm_mse_loss':
                align_loss = align_unnorm_mse_loss * self.align_mse_loss_weight
            elif self.align_loss == 'top1_nce_loss':
                align_loss = align_top1_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'topk_nce_loss':
                align_loss = align_topk_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'max_pos_nce_loss':
                align_loss = align_max_pos_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'min_pos_nce_loss':
                align_loss = align_min_pos_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'per_topk_nce_loss':
                align_loss = align_per_topk_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'kl_rank_loss':
                align_loss = align_kl_rank_loss * self.align_nce_loss_weight
            elif self.align_loss == 'mse_top1_nce_loss':
                align_loss = align_mse_loss * self.align_mse_loss_weight + align_top1_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'mse_topk_nce_loss':
                align_loss = align_mse_loss * self.align_mse_loss_weight + align_topk_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'mse_max_pos_nce_loss':
                align_loss = align_mse_loss * self.align_mse_loss_weight + align_max_pos_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'mse_min_pos_nce_loss':
                align_loss = align_mse_loss * self.align_mse_loss_weight + align_min_pos_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'mse_per_topk_nce_loss':
                align_loss = align_mse_loss * self.align_mse_loss_weight + align_per_topk_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'mse_kl_rank_loss':
                align_loss = align_mse_loss * self.align_mse_loss_weight + align_kl_rank_loss * self.align_nce_loss_weight
            elif self.align_loss == 'unnorm_mse_top1_nce_loss': 
                align_loss = align_unnorm_mse_loss * self.align_mse_loss_weight + align_top1_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'unnorm_mse_topk_nce_loss':
                align_loss = align_unnorm_mse_loss * self.align_mse_loss_weight + align_topk_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'unnorm_mse_max_pos_nce_loss':
                align_loss = align_unnorm_mse_loss * self.align_mse_loss_weight + align_max_pos_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'unnorm_mse_min_pos_nce_loss':
                align_loss = align_unnorm_mse_loss * self.align_mse_loss_weight + align_min_pos_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'unnorm_mse_per_topk_nce_loss':
                align_loss = align_unnorm_mse_loss * self.align_mse_loss_weight + align_per_topk_nce_loss * self.align_nce_loss_weight
            elif self.align_loss == 'unnorm_mse_kl_rank_loss':
                align_loss = align_unnorm_mse_loss * self.align_mse_loss_weight + align_kl_rank_loss * self.align_nce_loss_weight
            else:
                raise NotImplementedError(f"Not implemented alignment loss: {self.align_loss}")

            with torch.no_grad():
                # Top1 accuracy
                max_neg_scores, _ = masked_neg_scores.max(dim=1)
                align_top1_accuracy = (top1_pos_scores > max_neg_scores).float().mean()
                
                # Topk accuracy (all the top-k scores are larger than all negatives)
                min_pos_scores, _ = topk_pos_scores.min(dim=1)
                align_topk_accuracy = (min_pos_scores > max_neg_scores).float().mean()

                # Max positive accuracy (max score in the positive set is larger than all negatives)
                align_max_pos_accuracy = (max_pos_scores > max_neg_scores).float().mean()

                # cosine similarity
                align_cosine_similarity = (normalized_flat_subquery_embeddings * normalized_gen_embeddings).sum(dim=-1).mean()
                # Top1 cosine similarity
                align_top1_cosine_similarity = (top1_pos_scores * self.temperature).mean()
                # Topk cosine similarity
                align_topk_cosine_similarity = (topk_pos_scores * self.temperature).mean()
                # Max positive cosine similarity
                align_max_pos_cosine_similarity = (max_pos_scores * self.temperature).mean()   

            if not has_subquery:
                if self_decode_subquery_metrics is not None:
                    self_decode_subquery_metrics['num_elements'] = torch.tensor(
                        0, device=self_decode_subquery_metrics['num_elements'].device, dtype=torch.long)
                    self_decode_subquery_metrics['num_elements_without_spec_tokens'] = torch.tensor(
                        0, device=self_decode_subquery_metrics['num_elements_without_spec_tokens'].device, dtype=torch.long)
                align_num_elements = torch.tensor(0, device=align_num_elements.device, dtype=torch.long)
                
        gen_scaling_factor = 1.0
        gen_scaling_factor_without_spec_tokens = 1.0
        self_decode_thought_gen_scaling_factor = 1.0
        self_decode_thought_gen_scaling_factor_without_spec_tokens = 1.0
        self_decode_subquery_gen_scaling_factor = 1.0
        self_decode_subquery_gen_scaling_factor_without_spec_tokens = 1.0
        align_scaling_factor = 1.0
        if self.is_ddp:
            # gather num_elements for scaling results
            avg_gen_num_elements = self._dist_gather_scalar_mean(gen_num_elements)
            avg_gen_num_elements_without_spec_tokens = self._dist_gather_scalar_mean(gen_num_elements_without_spec_tokens)
            if self_decode_thought_metrics is not None:
                self_decode_thought_metrics['avg_num_elements'] = self._dist_gather_scalar_mean(self_decode_thought_metrics['num_elements'])
                self_decode_thought_metrics['avg_num_elements_without_spec_tokens'] = self._dist_gather_scalar_mean(self_decode_thought_metrics['num_elements_without_spec_tokens'])
            if self_decode_subquery_metrics is not None:
                self_decode_subquery_metrics['avg_num_elements'] = self._dist_gather_scalar_mean(self_decode_subquery_metrics['num_elements'])
                self_decode_subquery_metrics['avg_num_elements_without_spec_tokens'] = self._dist_gather_scalar_mean(self_decode_subquery_metrics['num_elements_without_spec_tokens'])
            if self.align_retriever:
                avg_align_num_elements = self._dist_gather_scalar_mean(align_num_elements)
            
            gen_scaling_factor = gen_num_elements / avg_gen_num_elements if avg_gen_num_elements > 0 else 0
            gen_scaling_factor_without_spec_tokens = gen_num_elements_without_spec_tokens / avg_gen_num_elements_without_spec_tokens \
                if avg_gen_num_elements_without_spec_tokens > 0 else 0
            if self_decode_thought_metrics is not None:
                self_decode_thought_gen_scaling_factor = self_decode_thought_metrics['num_elements'] / self_decode_thought_metrics['avg_num_elements'] \
                    if self_decode_thought_metrics['avg_num_elements'] > 0 else 0
                self_decode_thought_gen_scaling_factor_without_spec_tokens = self_decode_thought_metrics['num_elements_without_spec_tokens'] / self_decode_thought_metrics['avg_num_elements_without_spec_tokens'] \
                    if self_decode_thought_metrics['avg_num_elements_without_spec_tokens'] > 0 else 0
            if self_decode_subquery_metrics is not None:
                self_decode_subquery_gen_scaling_factor = self_decode_subquery_metrics['num_elements'] / self_decode_subquery_metrics['avg_num_elements'] \
                    if self_decode_subquery_metrics['avg_num_elements'] > 0 else 0
                self_decode_subquery_gen_scaling_factor_without_spec_tokens = self_decode_subquery_metrics['num_elements_without_spec_tokens'] / self_decode_subquery_metrics['avg_num_elements_without_spec_tokens'] \
                    if self_decode_subquery_metrics['avg_num_elements_without_spec_tokens'] > 0 else 0
            if self.align_retriever:
                align_scaling_factor = align_num_elements / avg_align_num_elements if avg_align_num_elements > 0 else 0

        # combine losses
        total_loss = self.gen_loss_weight*gen_loss*gen_scaling_factor
        if self.align_retriever:
            total_loss = total_loss + self.align_loss_weight*align_loss*align_scaling_factor
        if self_decode_thought_metrics is not None:
            total_loss = total_loss + self.self_decode_thought_gen_loss_weight*self_decode_thought_metrics['loss']*self_decode_thought_gen_scaling_factor
        if self_decode_subquery_metrics is not None:
            total_loss = total_loss + self.self_decode_subquery_gen_loss_weight*self_decode_subquery_metrics['loss']*self_decode_subquery_gen_scaling_factor
        
        # total_loss = align_loss  # for debug

        # TODO: for debug
        if total_loss.isnan() or total_loss.isinf():
            logger.error(f"total loss: {total_loss.item()}, gen_loss: {gen_loss.item()}, align_loss: {align_loss.item() if self.align_retriever else 'N/A'}, self_decode_thought_gen_loss: {self_decode_thought_metrics['loss'].item() if self_decode_thought_metrics is not None else 'N/A'}, self_decode_subquery_gen_loss: {self_decode_subquery_metrics['loss'].item() if self_decode_subquery_metrics is not None else 'N/A'}")
            logger.error(f"gen_num_elements: {gen_num_elements}, gen_num_elements_without_spec_tokens: {gen_num_elements_without_spec_tokens}, align_num_elements: {align_num_elements if self.align_retriever else 'N/A'}")
            logger.error(f"self_decode_thought_num_elements: {self_decode_thought_metrics['num_elements'] if self_decode_thought_metrics is not None else 'N/A'}, self_decode_thought_num_elements_without_spec_tokens: {self_decode_thought_metrics['num_elements_without_spec_tokens'] if self_decode_thought_metrics is not None else 'N/A'}")
            logger.error(f"self_decode_subquery_num_elements: {self_decode_subquery_metrics['num_elements'] if self_decode_subquery_metrics is not None else 'N/A'}, self_decode_subquery_num_elements_without_spec_tokens: {self_decode_subquery_metrics['num_elements_without_spec_tokens'] if self_decode_subquery_metrics is not None else 'N/A'}")
            logger.error(f"avg_gen_num_elements: {avg_gen_num_elements}, avg_gen_num_elements_without_spec_tokens: {avg_gen_num_elements_without_spec_tokens}, avg_align_num_elements: {avg_align_num_elements if self.align_retriever else 'N/A'}")
            logger.error(f"thought_gen_avg_num_elements: {self_decode_thought_metrics['avg_num_elements'] if self_decode_thought_metrics is not None else 'N/A'}, thought_gen_avg_num_elements_without_spec_tokens: {self_decode_thought_metrics['avg_num_elements_without_spec_tokens'] if self_decode_thought_metrics is not None else 'N/A'}")
            logger.error(f"subquery_gen_avg_num_elements: {self_decode_subquery_metrics['avg_num_elements'] if self_decode_subquery_metrics is not None else 'N/A'}, subquery_gen_avg_num_elements_without_spec_tokens: {self_decode_subquery_metrics['avg_num_elements_without_spec_tokens'] if self_decode_subquery_metrics is not None else 'N/A'}")
            raise ValueError(f"Total loss is NaN or Inf: {total_loss.item()}")

        return TrainingOutput(
            loss=total_loss,
            gen_loss=gen_loss*gen_scaling_factor,
            gen_accuracy=gen_accuracy*gen_scaling_factor,
            gen_accuracy_without_spec_tokens=gen_accuracy_without_spec_tokens*gen_scaling_factor_without_spec_tokens,
            self_decode_thought_gen_loss=self_decode_thought_metrics['loss']*self_decode_thought_gen_scaling_factor if self_decode_thought_metrics is not None else torch.tensor(torch.nan),
            self_decode_subquery_gen_loss=self_decode_subquery_metrics['loss']*self_decode_subquery_gen_scaling_factor if self_decode_subquery_metrics is not None else torch.tensor(torch.nan),
            self_decode_thought_gen_accuracy=self_decode_thought_metrics['accuracy']*self_decode_thought_gen_scaling_factor if self_decode_thought_metrics is not None else torch.tensor(torch.nan),
            self_decode_thought_gen_accuracy_without_spec_tokens=self_decode_thought_metrics['accuracy_without_spec_tokens']*self_decode_thought_gen_scaling_factor_without_spec_tokens if self_decode_thought_metrics is not None else torch.tensor(torch.nan),
            self_decode_subquery_gen_accuracy=self_decode_subquery_metrics['accuracy']*self_decode_subquery_gen_scaling_factor if self_decode_subquery_metrics is not None else torch.tensor(torch.nan),
            self_decode_subquery_gen_accuracy_without_spec_tokens=self_decode_subquery_metrics['accuracy_without_spec_tokens']*self_decode_subquery_gen_scaling_factor_without_spec_tokens if self_decode_subquery_metrics is not None else torch.tensor(torch.nan),
            align_loss=align_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_mse_loss=align_mse_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_unnorm_mse_loss=align_unnorm_mse_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_top1_nce_loss=align_top1_nce_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_topk_nce_loss=align_topk_nce_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_max_pos_nce_loss=align_max_pos_nce_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_min_pos_nce_loss=align_min_pos_nce_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_per_topk_nce_loss=align_per_topk_nce_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_kl_rank_loss=align_kl_rank_loss*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_top1_accuracy=align_top1_accuracy*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_topk_accuracy=align_topk_accuracy*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_max_pos_accuracy=align_max_pos_accuracy*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_cosine_similarity=align_cosine_similarity*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_top1_cosine_similarity=align_top1_cosine_similarity*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_topk_cosine_similarity=align_topk_cosine_similarity*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
            align_max_pos_cosine_similarity=align_max_pos_cosine_similarity*align_scaling_factor if self.align_retriever else torch.tensor(torch.nan),
        )

    def _dist_gather_scalar_mean(self, value: torch.Tensor):
        value_sum = value.clone()
        dist.all_reduce(value_sum, op=dist.ReduceOp.SUM)
        return value_sum.item() / self.world_size

    def _dist_gather_list(self, lst: Optional[List], return_offset=False):
        if lst is None:
            lst = []

        output = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(output, lst)
        
        flat_output = [item for sublist in output for item in sublist]

        if return_offset:
            offset = sum(len(sublist) for sublist in output[:self.process_rank])
            return flat_output, offset

        return flat_output

    def _dist_gather_tensor_variable_batch(self, t: Optional[torch.Tensor], return_offset=False):
        """Gather tensors with varying first dimension across distributed processes."""
        device = self.base_model.device
        
        if t is None:
            local_size = None
        else:
            t = t.contiguous()
            local_size = list(t.shape)
            
        size_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(size_list, local_size)
        
        if all(sz is None for sz in size_list):
            return None if not return_offset else (None, None)
        
        sizes = [sz[0] if sz is not None else 0 for sz in size_list]
        max_size = max(sizes)
        
        non_none_size_item = [sz for sz in size_list if sz is not None][0]
        padded_shape = [max_size] + non_none_size_item[1:]
        
        if t is not None:
            padding_size = [max_size - t.shape[0]] + list(t.shape[1:])
            padding = torch.zeros(padding_size, device=device, dtype=t.dtype)
            t_padded = torch.cat([t, padding], dim=0)
        else:
            t_padded = torch.zeros(padded_shape, device=device, dtype=torch.float)  # dummy tensor for gathering
            
        all_tensors = [torch.empty_like(t_padded) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t_padded)
        
        # truncate tensors back to original sizes
        if t is not None:
            all_tensors[self.process_rank] = t  # keep gradients
        # unpad tensors
        all_tensors = [tensor[:sizes[i]] for i, tensor in enumerate(all_tensors)]
        all_tensors = torch.cat(all_tensors, dim=0)
        
        if return_offset:
            offset = sum(sizes[:self.process_rank])
            return all_tensors, offset
        
        return all_tensors

    # inference
    def generate(self, infer_inputs: IterationRecord, batch_subquery_decoding=False) -> IterationRecord: 
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
            assert not infer_inputs.chunk_items, "LatentRAG should not have any retrieved chunks in the first turn."
            
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
        
        prompt += ''.join(self.spec_tokens['latent_tokens'])

        if cur_turn < self.max_turns: 
            thought_generation_start_time = time.perf_counter()
            
            next_token_id, action_choice_last_hidden_state, past_key_values, metadata = self._next_token_id_choice(
                prompt, past_key_values, 
                candidate_token_ids=[self.spec_token_ids['bosq_token'], self.spec_token_ids['boa_token']]
            )
            total_metadata['prompt_tokens'] += metadata['prompt_tokens']
            total_metadata['completion_tokens'] += metadata['completion_tokens']
            prompt = ''
            
            thought_generation_end_time = time.perf_counter()
            thought_generation_total_time = thought_generation_end_time - thought_generation_start_time
        else: 
            next_token_id = self.spec_token_ids['boa_token']
            infer_outputs.force_answer = True
            thought_generation_total_time = None

        if next_token_id == self.spec_token_ids['boa_token']:
            answer_generation_start_time = time.perf_counter()
            
            prompt += self.spec_tokens['boa_token']
            answer, answer_gen_input_last_hidden_state, past_key_values, metadata = self._text_generation(
                prompt, past_key_values, end_token_id=self.spec_token_ids['eoa_token'], 
                max_gen_tokens=self.max_gen_tokens
            )
            infer_outputs.answer = answer.strip()
            total_metadata['prompt_tokens'] += metadata['prompt_tokens']
            total_metadata['completion_tokens'] += metadata['completion_tokens']
            
            answer_generation_end_time = time.perf_counter()
            answer_generation_total_time = answer_generation_end_time - answer_generation_start_time
            subquery_generation_total_time = None
            
            # extract latent llm features
            num_latent_tokens = len(self.spec_token_ids['latent_tokens'])
            if infer_outputs.force_answer:
                latent_llm_feature = answer_gen_input_last_hidden_state[:, -num_latent_tokens-1:-1, :]
            else:
                latent_llm_feature = action_choice_last_hidden_state[:, -num_latent_tokens:, :]
            
            batch_latent_llm_features = infer_outputs.states.get('latent_llm_features', [])
            if batch_subquery_decoding:
                batch_latent_llm_features.append(latent_llm_feature)
            
            # batch subquery decoding
            if batch_subquery_decoding:
                batch_decoding_start_time = time.perf_counter()
                
                thoughts, subqueries, metadata = self._batch_thought_subquery_decoding(infer_outputs.states.get('latent_llm_features', []), infer_outputs.states.get('latent_subquery_llm_features', []))
                infer_outputs.thoughts = thoughts
                infer_outputs.subqueries = subqueries
                total_metadata['prompt_tokens'] += metadata['prompt_tokens']
                total_metadata['completion_tokens'] += metadata['completion_tokens']
                
                batch_decoding_end_time = time.perf_counter()
                batch_decoding_total_time = batch_decoding_end_time - batch_decoding_start_time
                
            infer_outputs.states = None
        else: 
            subquery_generation_start_time = time.perf_counter()
            
            num_latent_tokens = len(self.spec_token_ids['latent_tokens'])
            latent_llm_feature = action_choice_last_hidden_state[:, -num_latent_tokens:, :]
            
            prompt = ''.join(self.spec_tokens['latent_subquery_tokens'])
            num_subquery_latent_tokens = len(self.spec_token_ids['latent_subquery_tokens'])
            subquery_mask = torch.ones((1, num_subquery_latent_tokens), dtype=torch.bool, device=self.base_model.device)
            latent_subquery_llm_feature, past_key_values, metadata = self._get_forward_hidden_state_outputs(prompt, past_key_values)

            subquery_embedding = self._subquery_embedding_generation(
                latent_subquery_llm_feature
            )
            subquery_embedding = subquery_embedding.detach().cpu().numpy()
            infer_outputs.subquery_embeddings.append(subquery_embedding)
            
            subquery_generation_end_time = time.perf_counter()
            subquery_generation_total_time = subquery_generation_end_time - subquery_generation_start_time
            answer_generation_total_time = None
            
            batch_latent_llm_features = infer_outputs.states.get('latent_llm_features', [])
            if batch_subquery_decoding:
                batch_latent_llm_features.append(latent_llm_feature)
            batch_latent_subquery_llm_features = infer_outputs.states.get('latent_subquery_llm_features', [])
            if batch_subquery_decoding:
                batch_latent_subquery_llm_features.append(latent_subquery_llm_feature)
                
            infer_outputs.states = {
                "past_key_values": past_key_values, 
                "latent_llm_features": batch_latent_llm_features,
                "latent_subquery_llm_features": batch_latent_subquery_llm_features,
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
        if batch_subquery_decoding and 'batch_decoding_time' not in infer_outputs.extra_info['step_wise_time']:
            infer_outputs.extra_info['step_wise_time']['batch_decoding_time'] = []
        infer_outputs.extra_info['step_wise_time']['prefill_time'].append(prefill_total_time)
        infer_outputs.extra_info['step_wise_time']['thought_generation_time'].append(thought_generation_total_time)
        infer_outputs.extra_info['step_wise_time']['subquery_generation_time'].append(subquery_generation_total_time)
        infer_outputs.extra_info['step_wise_time']['answer_generation_time'].append(answer_generation_total_time)
        if batch_subquery_decoding and 'batch_decoding_total_time' in locals():
            infer_outputs.extra_info['step_wise_time']['batch_decoding_time'].append(batch_decoding_total_time)

        return infer_outputs, total_metadata
    
    def batch_generate(self, infer_inputs: List[IterationRecord], batch_subquery_decoding=False) -> List[IterationRecord]:

        if len(infer_inputs) > 1:
            logger.warning("Batch generation is not implemented! Will break the batch into individual requests.")
        infer_outputs = []
        total_metadata = {
            'prompt_tokens': 0,
            'completion_tokens': 0, 
            'num_request': len(infer_inputs)
        }
        for item in infer_inputs: 
            cur_infer_outputs, metadata = self.generate(item, batch_subquery_decoding=batch_subquery_decoding)
            infer_outputs.append(cur_infer_outputs)
            total_metadata['prompt_tokens'] += metadata['prompt_tokens']
            total_metadata['completion_tokens'] += metadata['completion_tokens']
        return infer_outputs, total_metadata

    def generate_with_step_wise_time(self, infer_inputs: List[IterationRecord], batch_subquery_decoding=False) -> List[IterationRecord]:
        assert len(infer_inputs) == 1, "Step-wise time measurement only supports batch size 1."
        
        return self.batch_generate(infer_inputs, batch_subquery_decoding=batch_subquery_decoding)
    
    def get_latent_logits(self, infer_inputs: IterationRecord, num=30):
        prompt = self.spec_prompts['query_prompt'].format(query=infer_inputs.query)
        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                add_generation_prompt=True, tokenize=False)
        
        for chunk_items in infer_inputs.reference_chunk_items:
            prompt += ''.join(self.spec_tokens['latent_tokens'])
            prompt += ''.join(self.spec_tokens['latent_subquery_tokens'])
            prompt += self.spec_tokens['bod_token']
            prompt += self._process_chunk_texts([chunk_item['text'] for chunk_item in chunk_items], 
                                                title_in_first_line=self.title_in_first_line)
            prompt += self.spec_tokens['eod_token']
        
        prompt += ''.join(self.spec_tokens['latent_tokens'])
        
        prompt_token_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(self.base_model.device)
        inputs_embeds = self._prompt_to_inputs_embeds(prompt_token_ids)
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            # output_hidden_states=True,
        )
        logits = outputs.logits
        # last_hidden_state = outputs.hidden_states[-1]
        
        # collect latent thought top logits
        latent_indices = (
            (prompt_token_ids == self.spec_token_ids['latent_end_token'])
        ).nonzero()
        num_latent_tokens = len(self.spec_token_ids['latent_tokens'])
        offsets = torch.arange(num_latent_tokens - 1, -1, -1, device=latent_indices.device)  # [0, ..., num_latent_tokens-1]
        token_indices = latent_indices[:, 1].unsqueeze(1) - offsets.unsqueeze(0)  # [N, num_latent_tokens]
        batch_indices = latent_indices[:, 0].unsqueeze(1).expand(-1, num_latent_tokens)
        thought_token_logits = logits[batch_indices, token_indices, :]
        thought_log_probs = torch.log_softmax(thought_token_logits, dim=-1)
        thought_probs = thought_log_probs.exp()
        thought_entropy = -(thought_probs * thought_log_probs).sum(dim=-1)  # [N, T]
        topk_logits, topk_ids = torch.topk(thought_token_logits, k=num, dim=-1)
        N, T, K = topk_ids.shape
        flat_ids = topk_ids.reshape(-1).tolist()
        flat_tokens = self.tokenizer.convert_ids_to_tokens(flat_ids)
        flat_tokens = [token.replace("Ġ", " ") for token in flat_tokens]
        topk_tokens = np.array(flat_tokens).reshape(N, T, K)
        topk_log_probs = torch.gather(thought_log_probs, dim=-1, index=topk_ids)
        topk_probs = topk_log_probs.exp()
        latent_thought_logits_info = {
            'topk_ids': topk_ids.detach().cpu().tolist(),
            'topk_tokens': topk_tokens.tolist(),
            'topk_logits': topk_logits.detach().cpu().tolist(),
            'topk_log_probs': topk_log_probs.detach().cpu().tolist(),
            'topk_probs': topk_probs.detach().cpu().tolist(),
            'entropy': thought_entropy.detach().cpu().tolist(),
        }
        
        # collect latent subquery top logits
        latent_subquery_indices = (
            (prompt_token_ids == self.spec_token_ids['latent_subquery_end_token'])
        ).nonzero()
        num_latent_subquery_tokens = len(self.spec_token_ids['latent_subquery_tokens'])
        if latent_subquery_indices.size(0) == 0:
            latent_subquery_logits_info = None
        else:
            offsets = torch.arange(num_latent_subquery_tokens - 1, -1, -1, device=latent_subquery_indices.device)  # [0, ..., num_latent_subquery_tokens-1]
            token_indices = latent_subquery_indices[:, 1].unsqueeze(1) - offsets.unsqueeze(0)  # [N, num_latent_subquery_tokens]
            batch_indices = latent_subquery_indices[:, 0].unsqueeze(1).expand(-1, num_latent_subquery_tokens)
            subquery_token_logits = logits[batch_indices, token_indices, :]
            subquery_log_probs = torch.log_softmax(subquery_token_logits, dim=-1)
            subquery_probs = subquery_log_probs.exp()
            subquery_entropy = -(subquery_probs * subquery_log_probs).sum(dim=-1)  # [N, T]
            topk_logits, topk_ids = torch.topk(subquery_token_logits, k=num, dim=-1)
            N, T, K = topk_ids.shape
            flat_ids = topk_ids.reshape(-1).tolist()
            flat_tokens = self.tokenizer.convert_ids_to_tokens(flat_ids)
            flat_tokens = [token.replace("Ġ", " ") for token in flat_tokens]
            topk_tokens = np.array(flat_tokens).reshape(N, T, K)
            topk_log_probs = torch.gather(subquery_log_probs, dim=-1, index=topk_ids)
            topk_probs = topk_log_probs.exp()
            latent_subquery_logits_info = {
                'topk_ids': topk_ids.detach().cpu().tolist(),
                'topk_tokens': topk_tokens.tolist(),
                'topk_logits': topk_logits.detach().cpu().tolist(),
                'topk_log_probs': topk_log_probs.detach().cpu().tolist(),
                'topk_probs': topk_probs.detach().cpu().tolist(),
                'entropy': subquery_entropy.detach().cpu().tolist(),
            }
            
        return latent_thought_logits_info, latent_subquery_logits_info

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
    def _prompt_to_inputs_embeds(self, prompt):
        if prompt is None or prompt == '':
            return None
        
        if isinstance(prompt, str): # text
            input_token_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).to(self.base_model.device)
            inputs_embeds = self.base_embedding_layer(input_token_ids)
        elif prompt.dtype == torch.long: # token ids
            inputs_embeds = self.base_embedding_layer(prompt)
        else: # embeddings
            inputs_embeds = prompt

        return inputs_embeds

    @torch.inference_mode()
    def _get_forward_hidden_state_outputs(self, input_data, past_key_values):
        inputs_embeds = self._prompt_to_inputs_embeds(input_data)
            
        metadata = dict()
        metadata['prompt_tokens'] = inputs_embeds.size(1)
        metadata['completion_tokens'] = 0
            
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-1]
        past_key_values = outputs.past_key_values

        return last_hidden_state, past_key_values, metadata

    @torch.inference_mode()
    def _next_token_id_choice(self, input_data, past_key_values, candidate_token_ids, output_full_hidden_state=False):
        inputs_embeds = self._prompt_to_inputs_embeds(input_data)
            
        metadata = dict()
        metadata['prompt_tokens'] = inputs_embeds.size(1)
        metadata['completion_tokens'] = 1

        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
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
    def _subquery_embedding_generation(self, latent_subquery_llm_feature):
        latent_subquery_llm_embeddings = self.llm_projector(latent_subquery_llm_feature)
        if self.llm_only_query_embedding: # for ablation
            embedding = latent_subquery_llm_embeddings[:, -1, :].float()
        else:
            embedding = self.encoder.gen_embeddings_with_latent_inputs(latent_subquery_llm_embeddings, is_query=True, return_tensors=True, no_grad=True).float() # use float32 for better stability
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)

        if self.retriever_use_fp16:
            embedding = embedding.to(torch.float16)
        else:
            embedding = embedding.to(torch.float32)

        return embedding
    
    @torch.inference_mode()
    def _batch_thought_subquery_decoding(self, latent_llm_features, latent_subquery_llm_features):
        total_metadata = {'prompt_tokens': 0, 'completion_tokens': 0}
        
        if len(latent_llm_features) == 0 and len(latent_subquery_llm_features) == 0:
            return [], [], total_metadata
        
        batch_size_thought = len(latent_llm_features)
        batch_size_subquery = len(latent_subquery_llm_features)
        
        if getattr(self, 'self_decode_thought_input_token_ids', None) is None or getattr(self, 'self_decode_subquery_input_token_ids', None) is None:
            prompts = [
                self.spec_prompts['self_decode_thought_prompt'], 
                self.spec_prompts['self_decode_subquery_prompt']
            ]
            if self.use_chat_template:
                prompts = self.tokenizer.apply_chat_template(
                    [[{"role": "user", "content": prompt}] for prompt in prompts], 
                    add_generation_prompt=True, tokenize=False)
            
            prompt_input_data = self.tokenizer(
                prompts,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors='pt',
            ).to(self.base_model.device)
            input_ids = prompt_input_data['input_ids']
            attention_mask = prompt_input_data['attention_mask']
            self.self_decode_thought_input_token_ids = input_ids[0].unsqueeze(0)
            self.self_decode_thought_attention_mask = attention_mask[0].unsqueeze(0)
            self.self_decode_subquery_input_token_ids = input_ids[1].unsqueeze(0)
            self.self_decode_subquery_attention_mask = attention_mask[1].unsqueeze(0)
            self.self_decode_thought_latent_begin_index = (self.self_decode_thought_input_token_ids[0] == self.spec_token_ids['latent_begin_token']).nonzero()[0].item()
            self.self_decode_subquery_latent_begin_index = (self.self_decode_subquery_input_token_ids[0] == self.spec_token_ids['latent_subquery_begin_token']).nonzero()[0].item()
            self.self_decode_thought_input_embeds_cache = self.base_embedding_layer(self.self_decode_thought_input_token_ids)
            self.self_decode_subquery_input_embeds_cache = self.base_embedding_layer(self.self_decode_subquery_input_token_ids)
            
        
        if batch_size_thought > 0:
            latent_llm_features = torch.cat(latent_llm_features, dim=0)
            latent_llm_self_decode_thought_embeddings = self.llm_self_thought_projector(latent_llm_features)

            # replace with latent_llm_self_decode_thought_embeddings
            self_decode_thought_inputs_embeds = self.self_decode_thought_input_embeds_cache
            self_decode_thought_inputs_embeds = self_decode_thought_inputs_embeds.repeat(batch_size_thought, 1, 1)
            num_latent_tokens = len(self.spec_token_ids['latent_tokens'])
            self_decode_thought_inputs_embeds[:, self.self_decode_thought_latent_begin_index:self.self_decode_thought_latent_begin_index+num_latent_tokens, :] = latent_llm_self_decode_thought_embeddings
            self_decode_thought_attention_mask = self.self_decode_thought_attention_mask.repeat(batch_size_thought, 1)
            
        if batch_size_subquery > 0:
            latent_subquery_llm_features = torch.cat(latent_subquery_llm_features, dim=0)
            latent_llm_self_decode_subquery_embeddings = self.llm_self_subquery_projector(latent_subquery_llm_features)

            # replace with latent_llm_self_decode_subquery_embeddings
            self_decode_subquery_inputs_embeds = self.self_decode_subquery_input_embeds_cache
            self_decode_subquery_inputs_embeds = self_decode_subquery_inputs_embeds.repeat(batch_size_subquery, 1, 1)
            num_latent_subquery_tokens = len(self.spec_token_ids['latent_subquery_tokens'])
            self_decode_subquery_inputs_embeds[:, self.self_decode_subquery_latent_begin_index:self.self_decode_subquery_latent_begin_index+num_latent_subquery_tokens, :] = latent_llm_self_decode_subquery_embeddings
            self_decode_subquery_attention_mask = self.self_decode_subquery_attention_mask.repeat(batch_size_subquery, 1)

        inputs_embeds = [self_decode_thought_inputs_embeds] if batch_size_thought > 0 else []
        inputs_embeds += [self_decode_subquery_inputs_embeds] if batch_size_subquery > 0 else []
        inputs_embeds = torch.cat(inputs_embeds, dim=0)
        attention_mask = [self_decode_thought_attention_mask] if batch_size_thought > 0 else []
        attention_mask += [self_decode_subquery_attention_mask] if batch_size_subquery > 0 else []
        attention_mask = torch.cat(attention_mask, dim=0)

        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=[self.spec_token_ids['eosq_token'], self.spec_token_ids['eot_token'], self.tokenizer.eos_token_id],
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=self.max_gen_tokens,
            do_sample=False,
            use_cache=True
        )
        completion_ids = outputs
        completion_texts = self.tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=False
        )
        
        # parse thougts and subqueries
        thoughts = []
        subqueries = []
        
        thought_completion_texts = completion_texts[:batch_size_thought]
        subquery_completion_texts = completion_texts[batch_size_thought:]
        
        for i, text in enumerate(thought_completion_texts):
            t = text.replace(self.tokenizer.pad_token, "").strip()
            thought = None
            if self.spec_tokens['bot_token'] in t and self.spec_tokens['eot_token'] in t:
                thought = t.rsplit(self.spec_tokens['bot_token'], 1)[1].rsplit(self.spec_tokens['eot_token'], 1)[0].strip()
                thoughts.append(thought)
            else:
                thoughts.append("")
                
        for i, text in enumerate(subquery_completion_texts):
            t = text.replace(self.tokenizer.pad_token, "").strip()
            subquery = None
            if self.spec_tokens['bosq_token'] in t and self.spec_tokens['eosq_token'] in t:
                subquery = t.rsplit(self.spec_tokens['bosq_token'], 1)[1].rsplit(self.spec_tokens['eosq_token'], 1)[0].strip()
                subqueries.append(subquery)
            else:
                subqueries.append("")
        
        total_metadata['prompt_tokens'] = inputs_embeds.size(1) * inputs_embeds.size(0)
        completion_lengths = []
        for ids in completion_ids:
            completion_lengths.append((ids != self.tokenizer.pad_token_id).sum().item())
        total_metadata['completion_tokens'] = sum(completion_lengths)
        
        return thoughts, subqueries, total_metadata
            
    
    @torch.inference_mode()
    def _text_generation(self, input_data, past_key_values, end_token_id=None,
                         max_gen_tokens=64, input_end_token=False, output_full_hidden_state=False):
        inputs_embeds = self._prompt_to_inputs_embeds(input_data)
        
        metadata = dict()
        metadata['prompt_tokens'] = inputs_embeds.size(1) if inputs_embeds is not None else 0
        metadata['completion_tokens'] = 0

        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
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
        
        # # tokens for query
        # spec_tokens['boq_token'] = '<query>'
        # spec_tokens['eoq_token'] = '</query>'

        # tokens for latent representation of the generated thought
        num_latent_thought_tokens = config.model.get('num_latent_thought_tokens', 8)
        spec_tokens['latent_tokens'] = [
            f'<latent_{i}>'  for i in range(num_latent_thought_tokens)
        ]
        spec_tokens['latent_begin_token'] = spec_tokens['latent_tokens'][0]
        spec_tokens['latent_end_token'] = spec_tokens['latent_tokens'][-1]
        
        # tokens for latent representation of the generated subquery
        num_latent_subquery_tokens = config.model.get('num_latent_subquery_tokens', 8)
        spec_tokens['latent_subquery_tokens'] = [
            f'<latent_subquery_{i}>'  for i in range(num_latent_subquery_tokens)
        ]
        spec_tokens['latent_subquery_begin_token'] = spec_tokens['latent_subquery_tokens'][0]
        spec_tokens['latent_subquery_end_token'] = spec_tokens['latent_subquery_tokens'][-1]
        
        # begin and end of decode thought/subquery
        spec_tokens['bot_token'] = '<think>'
        spec_tokens['eot_token'] = '</think>'
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
        # spec_prompts['query_prompt'] = spec_tokens['boq_token'] + "{query}" + spec_tokens['eoq_token']
        # spec_prompts['self_decode_thought_prompt'] = spec_tokens['bode_thought_token']  + ''.join(spec_tokens['latent_tokens']) + spec_tokens['eode_thought_token']
        # spec_prompts['self_decode_subquery_prompt'] = spec_tokens['bode_subquery_token'] + ''.join(spec_tokens['latent_subquery_tokens']) + spec_tokens['eode_subquery_token']
        
        spec_prompts['query_prompt'] = 'Answer the following question by reasoning step by step and retrieving necessary information at each step:\n' + "{query}\n"
        # spec_prompts['query_prompt'] = spec_tokens['boq_token'] + "{query}" + spec_tokens['eoq_token']
        spec_prompts['self_decode_thought_prompt'] = 'Decode the thought based on the latent representation: ' + ''.join(spec_tokens['latent_tokens'])
        spec_prompts['self_decode_subquery_prompt'] = 'Decode the subquery based on the latent representation: ' + ''.join(spec_tokens['latent_subquery_tokens'])

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

        encoder_model_config = AutoConfigSafeLoader.from_pretrained(config.retriever.encoder_name_or_path)
        embedding_size = encoder_model_config.hidden_size
        # projector between LLM outputs and retriever inputs
        llm_projector_type = config.model.get('projector_type', 'linear')
        llm_projector = build_projector(
            input_dim=base_model.config.hidden_size,
            output_dim=embedding_size,
            projector_type=llm_projector_type,
        )
        logger.info(f"llm_projector built with {count_parameters(llm_projector)} parameters.")
        
        decode_thought = getattr(config.custom_training, 'decode_thought', False)
        decode_subquery = getattr(config.custom_training, 'decode_subquery', False)
        llm_self_thought_projector = None
        llm_self_subquery_projector = None
        
        # self thought decoding projector
        llm_self_thought_projector_type = (config.model.get('self_thought_projector_type') or llm_projector_type)
        llm_self_thought_projector = build_projector(
            input_dim=base_model.config.hidden_size,
            output_dim=base_model.config.hidden_size,
            projector_type=llm_self_thought_projector_type,
        )
        logger.info(f"llm_self_thought_projector built with {count_parameters(llm_self_thought_projector)} parameters.")
        
        # self subquery decoding projector
        llm_self_subquery_projector_type = (config.model.get('self_subquery_projector_type') or llm_projector_type)
        llm_self_subquery_projector = build_projector(
            input_dim=base_model.config.hidden_size,
            output_dim=base_model.config.hidden_size,
            projector_type=llm_self_subquery_projector_type,
        )
        logger.info(f"llm_self_subquery_projector built with {count_parameters(llm_self_subquery_projector)} parameters.")

        # load `llm_projector` weights if exists
        if config.model.projector_path is not None:
            flag = load_model_weights(llm_projector, config.model.projector_path, 'llm_projector')
            assert flag, f"Failed to load llm_projector weights from {config.model.projector_path}"
            
        if config.model.self_thought_projector_path is not None:
            flag = load_model_weights(llm_self_thought_projector, config.model.self_thought_projector_path, 'llm_self_thought_projector')
            assert flag, f"Failed to load llm_self_thought_projector weights from {config.model.self_thought_projector_path}"
        
        if config.model.self_subquery_projector_path is not None:
            flag = load_model_weights(llm_self_subquery_projector, config.model.self_subquery_projector_path, 'llm_self_subquery_projector')
            assert flag, f"Failed to load llm_self_subquery_projector weights from {config.model.self_subquery_projector_path}"
            
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

        if not config.model.base_model_trainable:
            for param in base_model.parameters():
                param.requires_grad = False
        if not config.model.projector_trainable:
            for param in llm_projector.parameters():
                param.requires_grad = False
        if decode_thought and not config.model.self_thought_projector_trainable:
            for param in llm_self_thought_projector.parameters():
                param.requires_grad = False
        if decode_subquery and not config.model.self_subquery_projector_trainable:
            for param in llm_self_subquery_projector.parameters():
                param.requires_grad = False

        model = cls(
            base_model=base_model,
            llm_projector=llm_projector,
            llm_self_thought_projector=llm_self_thought_projector,
            llm_self_subquery_projector=llm_self_subquery_projector,
            tokenizer=tokenizer,
            config=config,
        )
            
        return model
    
    @classmethod
    def load(cls, config):
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
            attn_implementation=config.model.attn_implementation, 
            device_map="auto"
        )

        cls._resize_model_token_embeddings(base_model, tokenizer, config)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        encoder_model_config = AutoConfigSafeLoader.from_pretrained(config.retriever.encoder_name_or_path)
        embedding_size = encoder_model_config.hidden_size
        # projector between LLM outputs and retriever inputs
        llm_projector_type = config.model.get('projector_type', 'linear')
        llm_projector = build_projector(
            input_dim=base_model.config.hidden_size,
            output_dim=embedding_size,
            projector_type=llm_projector_type,
        )
        llm_projector = llm_projector.to(torch_dtype).cuda()
        
        llm_self_thought_projector = None
        llm_self_subquery_projector = None
        # self thought decoding projector
        llm_self_thought_projector_type = (config.model.get('self_thought_projector_type') or llm_projector_type)
        llm_self_thought_projector = build_projector(
            input_dim=base_model.config.hidden_size,
            output_dim=base_model.config.hidden_size,
            projector_type=llm_self_thought_projector_type,
        )
        llm_self_thought_projector = llm_self_thought_projector.to(torch_dtype).cuda()
        # self subquery decoding projector
        llm_self_subquery_projector_type = (config.model.get('self_subquery_projector_type') or llm_projector_type)
        llm_self_subquery_projector = build_projector(
            input_dim=base_model.config.hidden_size,
            output_dim=base_model.config.hidden_size,
            projector_type=llm_self_subquery_projector_type,
        )
        llm_self_subquery_projector = llm_self_subquery_projector.to(torch_dtype).cuda()

        # load `llm_projector` weights if exists
        if config.model.projector_path is not None:
            flag = load_model_weights(llm_projector, config.model.projector_path, 'llm_projector')
            assert flag, f"Failed to load llm_projector weights from {config.model.projector_path}"
            llm_projector = llm_projector.cuda()
            logger.info(f"llm_projector loaded with {count_parameters(llm_projector)} parameters.")
            
        if config.model.self_thought_projector_path is not None:
            flag = load_model_weights(llm_self_thought_projector, config.model.self_thought_projector_path, 'llm_self_thought_projector')
            assert flag, f"Failed to load llm_self_thought_projector weights from {config.model.self_thought_projector_path}"
            llm_self_thought_projector = llm_self_thought_projector.cuda()
            logger.info(f"llm_self_thought_projector loaded with {count_parameters(llm_self_thought_projector)} parameters.")

        if config.model.self_subquery_projector_path is not None:
            flag = load_model_weights(llm_self_subquery_projector, config.model.self_subquery_projector_path, 'llm_self_subquery_projector')
            assert flag, f"Failed to load llm_self_subquery_projector weights from {config.model.self_subquery_projector_path}"
            llm_self_subquery_projector = llm_self_subquery_projector.cuda()
            logger.info(f"llm_self_subquery_projector loaded with {count_parameters(llm_self_subquery_projector)} parameters.")

        if config.model.lora.lora_name_or_path:
            lora_config = LoraConfigSafeLoader.from_pretrained(
                config.model.lora.lora_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=config.model.attn_implementation,)
            lora_model = PeftModel.from_pretrained(base_model, config.model.lora.lora_name_or_path, config=lora_config, device_map="auto")
            lora_model = lora_model.merge_and_unload()
            logger.info(f"Loaded and merged LoRA weights from {config.model.lora.lora_name_or_path}")

            model = cls(
                base_model=lora_model,
                llm_projector=llm_projector,
                llm_self_thought_projector=llm_self_thought_projector,
                llm_self_subquery_projector=llm_self_subquery_projector,
                tokenizer=tokenizer,
                config=config
            )
        else:
            model = cls(
                base_model=base_model,
                llm_projector=llm_projector,
                llm_self_thought_projector=llm_self_thought_projector,
                llm_self_subquery_projector=llm_self_subquery_projector,
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

        # save adapter
        adapter_prefix = 'llm_projector.'
        adapter_state_dict = {k[len(adapter_prefix):]: v for k, v in state_dict.items() if k.startswith(adapter_prefix)}
        if save_safetensors:
            save_file(adapter_state_dict, os.path.join(output_dir, "llm_projector.safetensors"))
        else:
            torch.save(adapter_state_dict, os.path.join(output_dir, f'llm_projector.bin'))

        # save llm_self_thought_projector
        if self.llm_self_thought_projector is not None:
            llm_self_thought_projector_prefix = 'llm_self_thought_projector.'
            llm_self_thought_projector_state_dict = {k[len(llm_self_thought_projector_prefix):]: v for k, v in state_dict.items() if k.startswith(llm_self_thought_projector_prefix)}
            if save_safetensors:
                save_file(llm_self_thought_projector_state_dict, os.path.join(output_dir, "llm_self_thought_projector.safetensors"))
            else:
                torch.save(llm_self_thought_projector_state_dict, os.path.join(output_dir, f'llm_self_thought_projector.bin'))
            
        # save llm_self_subquery_projector
        if self.llm_self_subquery_projector is not None:
            llm_self_subquery_projector_prefix = 'llm_self_subquery_projector.'
            llm_self_subquery_projector_state_dict = {k[len(llm_self_subquery_projector_prefix):]: v for k, v in state_dict.items() if k.startswith(llm_self_subquery_projector_prefix)}
            if save_safetensors:
                save_file(llm_self_subquery_projector_state_dict, os.path.join(output_dir, "llm_self_subquery_projector.safetensors"))
            else:
                torch.save(llm_self_subquery_projector_state_dict, os.path.join(output_dir, f'llm_self_subquery_projector.bin'))
