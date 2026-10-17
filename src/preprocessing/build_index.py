import os
import glob
import re
import argparse
import jsonlines
from tqdm import tqdm
import faiss
import psutil, os
import gc
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from retriever.encoder import Encoder
from retriever.encoder_vllm import Encoder_vllm
from llm_tools.safe_loader import AutoConfigSafeLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Build Faiss Flat index for wiki18 corpus.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the wiki18 corpus file")
    parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for encoder")
    parser.add_argument("--model_path", type=str, required=True, help="Model path for encoder")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--encode_batch_size", type=int, default=4096, help="Batch size for encoding")
    parser.add_argument("--shard_size", type=int, default=1000000, help="Number of embeddings for each shard index")
    parser.add_argument("--use_vllm", action='store_true', help="Whether to use vllm for encoding")
    parser.add_argument("--shard_range_begin", type=int, default=None, help="Start shard index (inclusive)")
    parser.add_argument("--shard_range_end", type=int, default=None, help="End shard index (inclusive)")
    parser.add_argument("--save_fp16", action='store_true', help="Whether to save embeddings in FP16 format")

    args = parser.parse_args()
    
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        args.save_path = os.path.dirname(args.save_path) + "/DEBUG_" + os.path.basename(args.save_path)
    
    if args.use_vllm:
        # Ensure max_length does not exceed model's max_position_embeddings if available
        model_conf = AutoConfigSafeLoader.from_pretrained(args.model_path)
        max_position_embeddings = getattr(model_conf, 'max_position_embeddings', None)
        if max_position_embeddings is not None:
            if args.max_length is None or args.max_length > max_position_embeddings:
                args.max_length = max_position_embeddings
                logger.info(f"Setting max_length to model's max_position_embeddings: {args.max_length}")
    
    
    logger.info("Building index with the following parameters:")
    logger.info(str(args))

    # if os.path.exists(args.save_path) and any(os.scandir(args.save_path)):
    #     logger.info(f"Index already exists at {args.save_path}. Skipping index creation.")
    #     exit(0)
    os.makedirs(args.save_path, exist_ok=True)

    pid = os.getpid()
    p = psutil.Process(pid)
    logger.info(f"Initial memory usage: {p.memory_info().rss / 1024 ** 2:.2f} MB")

    config_dict = {
        "training": None,
        "retriever": {
            "retriever_tag": args.model_name,
            "encoder_name_or_path": args.model_path,
            "tokenizer_name_or_path": None, 
            "encoder_query_max_length": args.max_length,
            "attn_implementation": "sdpa"
        }, 
        "vllm": {
            "encoder_model_gpu_memory_utilization": 0.83,
            "encoder_model_max_model_len": args.max_length,
            "encoder_model_max_num_seqs": args.encode_batch_size,
        }
    }
    config = OmegaConf.create(config_dict)
    if args.use_vllm:
        encoder = Encoder_vllm.load(config, silent=False)
    else:
        encoder = Encoder.load(config, silent=False)
    encoder.eval()

    p = psutil.Process(pid)
    logger.info(f"Initial memory after initializing encoder: {p.memory_info().rss / 1024 ** 2:.2f} MB")

    with jsonlines.open(args.file_path) as reader:
        corpus = [obj['contents'] for obj in tqdm(reader, desc="Loading corpus")]
    logger.info(f"Memory after loading corpus: {p.memory_info().rss / 1024 ** 2:.2f} MB")

    num_shards = (len(corpus) + args.shard_size - 1) // args.shard_size
    logger.info(f"Total corpus size: {len(corpus)}, number of shards to create: {num_shards}")
    for i, begin_idx in enumerate(range(0, len(corpus), args.shard_size)):
        if args.shard_range_begin is not None and i < args.shard_range_begin:
            logger.info(f"Skipping shard {i} as it is before shard_range_begin {args.shard_range_begin}")
            continue
        
        if args.shard_range_end is not None and i > args.shard_range_end:
            logger.info(f"Terminating encoding as shard {i} is after shard_range_end {args.shard_range_end}")
            break
        
        shard_save_path = os.path.join(args.save_path, f'shard_index_{i:04d}.npy')
        if os.path.exists(shard_save_path):
            logger.info(f"Shard index already exists at {shard_save_path}. Skipping this shard.")
            continue
        
        end_idx = min(begin_idx + args.shard_size, len(corpus))
        logger.info(f"Processing shard {i}: {begin_idx} to {end_idx}")
        shard_data = corpus[begin_idx:end_idx]
        
        if args.use_vllm:
            all_embeddings = encoder.single_batch_encode(shard_data, is_query=False, 
                                                         return_tensors=False)
        else:
            all_embeddings = encoder.multi_gpu_encode(shard_data, batch_size=args.encode_batch_size, is_query=False, 
                                                      return_tensors=False)
            
        if args.save_fp16:
            all_embeddings = all_embeddings.astype(np.float16)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        np.save(shard_save_path, all_embeddings)
        
        del all_embeddings
        gc.collect()
        logger.info(f"Memory after processing shard {i}: {p.memory_info().rss / 1024 ** 2:.2f} MB")
if __name__ == '__main__':
    main()
