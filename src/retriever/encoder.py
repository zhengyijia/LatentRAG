import os
import re
import numpy as np
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import torch
from torch import nn
import logging
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from packaging import version
import transformers

from llm_tools.safe_loader import AutoModelSafeLoader, AutoTokenizerSafeLoader, AutoConfigSafeLoader, LoraConfigSafeLoader
from utils import count_parameters

logger = logging.getLogger(__name__)

def check_transformers_version(model_name):
    required_versions = {
        "jina-embeddings-v5-text-nano": "4.40.0",
    }
    
    required_version = None
    for key in required_versions.keys():
        if key in model_name.lower():
            required_version = required_versions[key]
            break
    
    if required_version is None:
        return  # No specific version requirement for this model

    if version.parse(transformers.__version__) >= version.parse(required_version):
        return
    else:
        raise RuntimeError(f"Transformers >= {required_version} required for model {model_name}, "
                           f"but found {transformers.__version__}")

def load_model(model_path: str, tokenizer_path=None, dtype=torch.float32, dropout=0.0,
               attn_implementation=None, device_map=None):
    tokenizer_path = tokenizer_path if tokenizer_path is not None else model_path
    
    # set dropout
    config = AutoConfigSafeLoader.from_pretrained(model_path)
    dropout_keys = [
        # e5-base-v2
        "attention_probs_dropout_prob",
        "hidden_dropout_prob",
        "classifier_dropout",
        # qwen3-embedding
        "attention_dropout"
    ]
    for key in dropout_keys:
        if hasattr(config, key):
            setattr(config, key, dropout)
    
    model = AutoModelSafeLoader.from_pretrained(
        model_path, config=config, 
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map
    )
    
    if "jina-embeddings-v5" in model_path.lower():
        model.set_adapter("retrieval") # only use the retrieval adapter
        model = model.merge_and_unload()
    
    if device_map is None:
        model.cuda()
    # model = model.to(dtype)
    tokenizer = AutoTokenizerSafeLoader.from_pretrained(tokenizer_path, use_fast=True)
    
    vocab_size = len(tokenizer)
    embedding_layer = model.get_input_embeddings()
    num_embeddings = embedding_layer.num_embeddings
    if num_embeddings != vocab_size:
        logger.warning(
            "Vocab size mismatch: "
            f"Model from {model_path} has vocab size {num_embeddings}, "
            f"but tokenizer from {tokenizer_path} has vocab size {vocab_size}. "
            "Resizing model embeddings!!!")
        model.resize_token_embeddings(vocab_size)
    
    logger.info(f"Retriever model loaded with {count_parameters(model)} parameters.")
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean", 
    model=None
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]  # Careful: ensure the tokenizer uses right padding
    elif pooling_method == "last":
        return last_hidden_state[:, -1]  # Careful: ensure the tokenizer uses left padding
    elif pooling_method == "pooler":
        return pooler_output
    elif pooling_method == "custom":
        features = {
            "token_embeddings": last_hidden_state,
            "attention_mask": attention_mask
        }
        pooling_modules = [model[i] for i in range(1, len(model))]
        for pooling_module in pooling_modules:
            features = pooling_module(features)
        return features['sentence_embedding']
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder(nn.Module):
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
                 trainable=False,
                 silent=False):
        super().__init__()
        self.model_name = model_name
        self.base_model = base_model
        self.base_embedding_layer = base_model.get_input_embeddings()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.trainable = trainable
        self.silent = silent
        self.gpu_num = torch.cuda.device_count()
        self.embedding_size = self.base_model.config.hidden_size

        if "qwen3-embedding" in self.model_name.lower():
            self.pooling_method = "last"
        elif "e5" in self.model_name.lower():
            self.pooling_method = "mean"
        elif "jina-embeddings-v5" in self.model_name.lower():
            self.pooling_method = "last"
        elif "harrier-oss-v1" in self.model_name.lower():
            self.pooling_method = "last"
        elif "f2llm-v2" in self.model_name.lower():
            self.pooling_method = "last"
        
        if self.pooling_method == 'cls' and self.tokenizer.padding_side != 'right':
            logger.warning("For 'cls' pooling, tokenizer.padding_side should be 'right'. Resetting it to 'right'!")
            self.tokenizer.padding_side = 'right'
        elif self.pooling_method == 'last' and self.tokenizer.padding_side != 'left':
            logger.warning("For 'last' pooling, tokenizer.padding_side should be 'left'. Resetting it to 'left'!")
            self.tokenizer.padding_side = 'left'
        
        self.save_embedding_layers = False
        
        # freeze the base encoder model
        if not trainable:
            for p in self.base_model.parameters():
                p.requires_grad = False
                
    def update_tokenizer(self, new_tokenizer):
        old_tokenizer_name = getattr(self.tokenizer, 'name_or_path', 'unknown')
        new_tokenizer_name = getattr(new_tokenizer, 'name_or_path', 'unknown')
        logger.info(f"Updating tokenizer from {old_tokenizer_name} to {new_tokenizer_name}...")
        
        base_model = self.base_model
        old_emb = base_model.get_input_embeddings().weight.data
        hidden_size = old_emb.size(1)

        old_vocab = self.tokenizer.get_vocab()
        new_vocab = new_tokenizer.get_vocab()

        old_unk_id = self.tokenizer.unk_token_id

        new_emb = torch.zeros(len(new_vocab), hidden_size)

        def clean_token(token: str):
            return token.replace("Ġ", "").replace("▁", "")

        new_tokens = list(new_vocab.keys())
        new_tokens_clean = [clean_token(t) for t in new_tokens]
        batch_sub_tokens = self.tokenizer(
            new_tokens_clean,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        for token, new_id, sub_ids in zip(new_tokens, new_vocab.values(), batch_sub_tokens):
            # exact match
            if token in old_vocab:
                new_emb[new_id] = old_emb[old_vocab[token]]
                continue

            # filter out unk_id
            sub_ids = [i for i in sub_ids if i != old_unk_id]
            if len(sub_ids) == 0:
                sub_ids = [old_unk_id]

            sub_emb = old_emb[sub_ids]
            new_emb[new_id] = sub_emb.mean(dim=0)

        # update base_model and tokenizer
        base_model.resize_token_embeddings(len(new_tokenizer))
        base_model.get_input_embeddings().weight.data = new_emb
        self.tokenizer = new_tokenizer
        
        self.save_embedding_layers = True
        
        logger.info(f"Tokenizer updated. New vocab size: {len(new_vocab)}. Embedding layer resized and initialized.")
        
    def get_embedding_size(self):
        return self.embedding_size

    def get_base_embedding_layer(self):
        return self.base_embedding_layer

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
        elif "jina-embeddings-v5" in self.model_name.lower():
            if is_query:
                query_list = [f'Query: {query}' for query in query_list]
            else:
                query_list = [f'Document: {query}' for query in query_list]
        elif "harrier-oss-v1" in self.model_name.lower():
            if is_query:
                query_list = [f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}' for query in query_list]
        elif "f2llm-v2" in self.model_name.lower():
            if is_query:
                query_list = [f'Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: {query}' for query in query_list]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        
        return inputs

    def gen_embeddings_with_latent_inputs(self, latent_embeds: torch.Tensor, is_query=True, return_tensors=False, no_grad=True):
        prefix_attr = "query_prefix_inputs_embeds" if is_query else "doc_prefix_inputs_embeds"
        prefix_inputs_embeds = None

        # When the encoder is trainable, cached prefix embeddings would keep a stale
        # autograd graph across steps and can trigger "backward through the graph a
        # second time". Recompute per call in that case.
        if self.trainable:
            prefix_input_ids = self.format_inputs('', is_query=is_query)["input_ids"]
            prefix_inputs_embeds = self.base_embedding_layer(prefix_input_ids)
        else:
            if not hasattr(self, prefix_attr) or getattr(self, prefix_attr) is None:
                prefix_input_ids = self.format_inputs('', is_query=is_query)["input_ids"]
                cached_prefix = self.base_embedding_layer(prefix_input_ids).detach()
                setattr(self, prefix_attr, cached_prefix)
            prefix_inputs_embeds = getattr(self, prefix_attr)
        
        inputs_embeds = prefix_inputs_embeds.expand(
            latent_embeds.size(0), -1, -1
        )
        if "qwen3-embedding" in self.model_name.lower() \
            or "e5" in self.model_name.lower() \
            or "jina-embeddings-v5-text-nano" in self.model_name.lower() \
            or "harrier-oss-v1" in self.model_name.lower() \
            or "f2llm-v2" in self.model_name.lower():
            inputs_embeds = torch.cat([
                inputs_embeds[:, :-1, :], 
                latent_embeds,
                inputs_embeds[:, -1:, :] # the last eos token
            ], dim=1)
        else:
            raise NotImplementedError("Model is not implemented in `format_inputs_with_latent_embeds`")
        
        attention_mask = torch.ones(
            inputs_embeds.shape[:-1], dtype=torch.long
        ).to(inputs_embeds.device)
        inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask
        }
        return self.gen_embeddings(inputs, return_tensors=return_tensors, no_grad=no_grad)

    def gen_embeddings(self, inputs, return_tensors=False, no_grad=True):
        transformer_model = self.base_model

        if no_grad:
            with torch.no_grad():
                output = transformer_model(**inputs, return_dict=True)
        else:
            output = transformer_model(**inputs, return_dict=True)
        pooler_output = output.get("pooler_output", None)
        last_hidden_state = output.get("last_hidden_state", None)
        query_emb = pooling(pooler_output, last_hidden_state, inputs["attention_mask"], self.pooling_method, model=self.base_model)

        if not return_tensors:
            if query_emb.dtype == torch.bfloat16:
                query_emb = query_emb.to(torch.float32)
            query_emb = query_emb.detach().cpu().numpy()

        return query_emb

    def single_batch_encode(self, query_list: Union[List[str], str], is_query=True, return_tensors=False, no_grad=True) -> np.ndarray:
        
        inputs = self.format_inputs(query_list, is_query)
        query_emb = self.gen_embeddings(inputs, return_tensors=return_tensors, no_grad=no_grad)

        return query_emb

    def encode(self, query_list: List[str], batch_size=64, is_query=True, 
               return_tensors=False, no_grad=True) -> np.ndarray:
        query_emb = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: ", disable=self.silent):
            emb = self.single_batch_encode(query_list[i : i + batch_size], is_query, 
                                                      return_tensors=return_tensors, no_grad=no_grad)
            query_emb.append(emb)
            torch.cuda.empty_cache()
            
        if not return_tensors:
            query_emb = np.concatenate(query_emb, axis=0)
        else:
            query_emb = torch.cat(query_emb, dim=0)
        return query_emb

    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=64, is_query=True,
                         return_tensors=False) -> np.ndarray:
        if self.gpu_num > 1:
            is_hf_sharded = hasattr(self.base_model, "hf_device_map")
            if not is_hf_sharded and not isinstance(self.base_model, torch.nn.DataParallel):
                self.base_model = torch.nn.DataParallel(self.base_model)
        
        query_emb = self.encode(query_list, batch_size, is_query,
                                return_tensors=return_tensors)
        return query_emb
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable()
    
    @classmethod
    def build(cls, config, trainable=False, use_lora=False, custom_encoder=False):
        check_transformers_version(config.retriever.encoder_name_or_path)
        
        torch_dtype = torch.float32
        if getattr(config.training, 'bf16', False): 
            torch_dtype = torch.bfloat16
        elif getattr(config.training, 'fp16', False):
            torch_dtype = torch.float16
        
        base_model_dropout = getattr(config.retriever, "dropout", 0.0)
        
        if custom_encoder and getattr(config.retriever, "custom_encoder_name_or_path", None):
            try:
                base_model, tokenizer = load_model(
                    config.retriever.custom_encoder_name_or_path, 
                    tokenizer_path=config.retriever.custom_tokenizer_name_or_path,
                    dtype=torch_dtype, dropout=base_model_dropout, 
                    attn_implementation=config.retriever.attn_implementation)
                logger.info(f"Loaded model: {config.retriever.custom_encoder_name_or_path}")
            except Exception as e:
                logger.warning(f"Failed to load {config.retriever.custom_encoder_name_or_path}: {e}")
                logger.warning(f"Falling back to default model: {config.retriever.encoder_name_or_path}")
                base_model, tokenizer = load_model(
                    config.retriever.encoder_name_or_path, 
                    tokenizer_path=config.retriever.tokenizer_name_or_path,
                    dtype=torch_dtype, dropout=base_model_dropout, 
                    attn_implementation=config.retriever.attn_implementation)
        else:
            base_model, tokenizer = load_model(
                config.retriever.encoder_name_or_path, 
                tokenizer_path=config.retriever.tokenizer_name_or_path,
                dtype=torch_dtype, dropout=base_model_dropout, 
                attn_implementation=config.retriever.attn_implementation)
            logger.info(f"Loaded model: {config.retriever.encoder_name_or_path}")
        
        if use_lora:
            if config.retriever.encoder_lora.lora_name_or_path:
                lora_config = LoraConfigSafeLoader.from_pretrained(
                    config.retriever.encoder_lora.lora_name_or_path, 
                    attn_implementation=config.retriever.attn_implementation,
                )
                lora_model = PeftModel.from_pretrained(base_model, config.retriever.encoder_lora.lora_name_or_path, is_trainable=True)
                logger.info(f"Loaded LoRA weights from {config.retriever.encoder_lora.lora_name_or_path}")
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=config.retriever.encoder_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=config.retriever.encoder_lora.lora_r,
                    lora_alpha=config.retriever.encoder_lora.lora_alpha,
                    lora_dropout=config.retriever.encoder_lora.lora_dropout,
                    target_modules=config.retriever.encoder_lora.lora_target_modules.split(','),
                    inference_mode=False, 
                    use_rslora=config.retriever.encoder_lora.use_rslora,
                )
                lora_model = get_peft_model(base_model, lora_config)
            
            logger.info("Encoder model trainable parameters:")    
            lora_model.print_trainable_parameters()

            encoder = cls(model_name=config.retriever.retriever_tag,
                          base_model=lora_model,
                          tokenizer=tokenizer,
                          max_length=config.retriever.encoder_query_max_length,
                          trainable=trainable,
                          silent=True)
        else:
            encoder = cls(model_name=config.retriever.retriever_tag,
                          base_model=base_model,
                          tokenizer=tokenizer,
                          max_length=config.retriever.encoder_query_max_length,
                          trainable=trainable,
                          silent=True)
        return encoder
    
    @classmethod
    def load(cls, config, trainable=False, use_lora=False, silent=True, custom_encoder=False):
        check_transformers_version(config.retriever.encoder_name_or_path)
        
        torch_dtype = torch.float32
        if getattr(config.training, 'bf16', False): 
            torch_dtype = torch.bfloat16
        elif getattr(config.training, 'fp16', False):
            torch_dtype = torch.float16
        
        device_map = "auto" if "jina-embeddings-v5" not in config.retriever.encoder_name_or_path.lower() else None
        
        if custom_encoder and getattr(config.retriever, "custom_encoder_name_or_path", None):
            try:
                base_model, tokenizer = load_model(
                    config.retriever.custom_encoder_name_or_path, 
                    tokenizer_path=getattr(config.retriever, "custom_tokenizer_name_or_path", None),
                    dtype=torch_dtype, 
                    attn_implementation=config.retriever.attn_implementation,
                    device_map=device_map)
                logger.info(f"Loaded model: {config.retriever.custom_encoder_name_or_path}")
            except Exception as e:
                logger.warning(f"Failed to load {config.retriever.custom_encoder_name_or_path}: {e}")
                logger.warning(f"Falling back to default model: {config.retriever.encoder_name_or_path}")
                base_model, tokenizer = load_model(
                    config.retriever.encoder_name_or_path, 
                    tokenizer_path=getattr(config.retriever, "tokenizer_name_or_path", None),
                    dtype=torch_dtype, 
                    attn_implementation=config.retriever.attn_implementation,
                    device_map=device_map)
        else:
            base_model, tokenizer = load_model(
                config.retriever.encoder_name_or_path, 
                tokenizer_path=getattr(config.retriever, "tokenizer_name_or_path", None),
                dtype=torch_dtype, 
                attn_implementation=config.retriever.attn_implementation,
                device_map=device_map)
            logger.info(f"Loaded model: {config.retriever.encoder_name_or_path}")
        
        if use_lora and config.retriever.encoder_lora.lora_name_or_path:
            lora_config = LoraConfigSafeLoader.from_pretrained(
                config.retriever.encoder_lora.lora_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=config.retriever.attn_implementation,)
            lora_model = PeftModel.from_pretrained(base_model, config.retriever.encoder_lora.lora_name_or_path, config=lora_config, device_map=device_map)
            lora_model = lora_model.merge_and_unload()
            logger.info(f"Loaded LoRA weights from {config.retriever.encoder_lora.lora_name_or_path}")

            encoder = cls(model_name=config.retriever.retriever_tag,
                          base_model=lora_model,
                          tokenizer=tokenizer,
                          max_length=config.retriever.encoder_query_max_length,
                          trainable=trainable,
                          silent=silent)
        else:
            encoder = cls(model_name=config.retriever.retriever_tag,
                          base_model=base_model,
                          tokenizer=tokenizer,
                          max_length=config.retriever.encoder_query_max_length,
                          trainable=trainable,
                          silent=silent)
        
        return encoder
    
    def save(self, output_dir: str, state_dict=None, save_safetensors: bool=True):
        if not self.trainable:
            return
        
        if state_dict is None:
            state_dict = self.state_dict()      

        base_prefix = 'base_model.'
        # assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        base_state_dict = {k[len(base_prefix):]: v for k, v in state_dict.items() if k.startswith(base_prefix)}
        
        self.base_model.save_pretrained(
            output_dir, state_dict=base_state_dict, safe_serialization=save_safetensors, 
            save_embedding_layers=self.save_embedding_layers
        )
        
        if getattr(self, 'tokenizer', None):
            self.tokenizer.save_pretrained(output_dir)
