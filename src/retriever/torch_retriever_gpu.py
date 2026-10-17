# modified from Search-R1
# https://github.com/PeterGriffinJin/Search-R1/blob/e23b87911693e5b174fc4d2d6bfd21f99137b5fc/search_r1/search/retrieval_server.py

import os
import glob
import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import torch
import logging
import time
import jsonlines
import gc
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class ChunkItem:
    chunk_id: str
    text: str
    score: float

@dataclass
class RetrieverOutput:
    results: List[List[ChunkItem]]
    start_time: Optional[float] = None
    end_time: Optional[float] = None

def load_corpus(corpus_path: str):
    with jsonlines.open(corpus_path) as reader:
        corpus = [obj for obj in tqdm(reader, desc="Loading corpus")]
    # corpus = datasets.load_dataset(
    #     'json', 
    #     data_files=corpus_path,
    #     split="train",
    #     num_proc=16
    # )
    
    return corpus

def load_chunks(corpus, chunk_ids):
    results = [corpus[chunk_id]['contents'] for chunk_id in chunk_ids]
    return results

class TorchInnerProductIndex:
    def __init__(self, device='cpu', use_fp16=True):
        self.index = None
        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.ntotal = 0

    @torch.inference_mode()
    def add(self, vectors):
        vectors = torch.from_numpy(vectors).contiguous()
        vectors = vectors.to(self.dtype).to(self.device)
        if self.index is None:
            self.index = vectors
        else:
            self.index = torch.cat([self.index, vectors], dim=0)
            
        self.ntotal = self.index.shape[0]
        
        torch.cuda.empty_cache()
        
    @torch.inference_mode()
    def search(self, query_vectors, k):
        if self.index is None or self.index.shape[0] == 0:
            empty_scores = np.full((query_vectors.shape[0], k), -np.inf, dtype=np.float32)
            empty_indices = np.full((query_vectors.shape[0], k), -1, dtype=np.int64)
            return empty_scores, empty_indices

        actual_k = min(k, self.index.shape[0])
        query_vectors = torch.from_numpy(query_vectors).to(self.dtype).to(self.device)
        # compute inner product
        scores = torch.matmul(query_vectors, self.index.T)
        top_scores, top_indices = torch.topk(scores, k=actual_k, dim=1)
        top_scores = top_scores.cpu().numpy().astype(np.float32)
        top_indices = top_indices.cpu().numpy().astype(np.int64)

        if actual_k < k:
            pad_scores = np.full((query_vectors.shape[0], k - actual_k), -np.inf, dtype=np.float32)
            pad_indices = np.full((query_vectors.shape[0], k - actual_k), -1, dtype=np.int64)
            top_scores = np.concatenate([top_scores, pad_scores], axis=1)
            top_indices = np.concatenate([top_indices, pad_indices], axis=1)

        torch.cuda.empty_cache()

        return top_scores, top_indices

class IndexShards:
    def __init__(self):
        self.shard_indexes = []
        self.offsets = []
        self.ntotal = 0
        
    @torch.inference_mode()
    def add(self, shard_index):
        self.offsets.append(self.ntotal)
        self.shard_indexes.append(shard_index)
        self.ntotal += shard_index.ntotal
        
    @torch.inference_mode()
    def search_one(self, shard, offset, query, k):
        scores, indices = shard.search(query, k)
        return (scores, indices + offset)
        
    @torch.inference_mode()
    def search(self, query_vectors, k):        
        with ThreadPoolExecutor(len(self.shard_indexes)) as ex:
            results = list(ex.map(
                lambda x: self.search_one(*x, query_vectors, k),
                [(s, o) for s, o in zip(self.shard_indexes, self.offsets)]
            ))
            
        all_scores = np.concatenate([r[0] for r in results], axis=1)
        all_indices = np.concatenate([r[1] for r in results], axis=1)

        # all_scores = []
        # all_indices = []        
        # for shard_index, offset in zip(self.shard_indexes, self.offsets):
        #     scores, indices = shard_index.search(query_vectors, k)
        #     all_scores.append(scores)
        #     all_indices.append(indices + offset)  # adjust indices by offset
        # all_scores = np.concatenate(all_scores, axis=1)  # concatenate along the index dimension
        # all_indices = np.concatenate(all_indices, axis=1)
        
        # final topk
        idx = np.argpartition(-all_scores, k-1, axis=1)[:, :k]
        top_scores = np.take_along_axis(all_scores, idx, axis=1)
        top_indices = np.take_along_axis(all_indices, idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        top_indices = np.take_along_axis(top_indices, order, axis=1)
        
        torch.cuda.empty_cache()
        
        return top_scores, top_indices

class EmptyRetriever():
    def __init__(self, config):
        self.config = config

    def get_retriever_tag(self):
        return 'none'

    def retrieve_with_text(self, queries, top_k=0):
        if top_k != 0:
            raise ValueError("EmptyRetriever only supports top_k=0")
        
        if isinstance(queries, str):
            queries = [queries]
        return RetrieverOutput(
            results=[[] for _ in range(len(queries))], 
            start_time=time.perf_counter(),
            end_time=time.perf_counter()
        )
        
    def retrieve_with_embeddings(self, query_embeddings, top_k=0):
        if top_k != 0:
            raise ValueError("EmptyRetriever only supports top_k=0")

        return RetrieverOutput(
            results=[[] for _ in range(len(query_embeddings))],
            start_time=time.perf_counter(),
            end_time=time.perf_counter()
        )

class TorchRetriever(): 

    def __init__(self, config):
        self.retriever_tag = config.retriever.retriever_tag
        
        # if config.retriever.encoder_name_or_path is not None:
        #     self.encoder = Encoder.load(config)
        #     self.encoder.eval()

        self.corpus_index_folder = os.path.join(config.retriever.corpus_path,
                                                config.retriever.corpus_index_folder)
        logger.info(f"Loading index from {self.corpus_index_folder}")
        all_paths = glob.glob(os.path.join(self.corpus_index_folder, "shard_index_*.npy"))
        shard_paths = [p for p in all_paths if re.search(r"shard_index_\d{4}\.npy$", p)]
        assert len(shard_paths) > 0
        
        def extract_number(path):
            match = re.search(r"shard_index_(\d{4})\.npy$", path)
            return int(match.group(1)) if match else -1
        shard_paths.sort(key=extract_number)

        if not config.retriever.faiss_gpu:
            self.index = TorchInnerProductIndex(device='cpu', use_fp16=config.retriever.use_fp16)
            for path in tqdm(shard_paths, desc="Loading torch shard indexes"):
                embeddings = np.load(path).astype(np.float32)
                # normalize
                embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
                self.index.add(embeddings)
                
                del embeddings
                gc.collect()
        else:
            n_gpu = torch.cuda.device_count()
            logger.info(f"Number of available GPUs: {n_gpu}")
            
            # num index
            n_indexes = n_gpu * 2  # Set to larger than 1 to avoid OOM when constructing index on GPU
            # n_indexes = n_gpu
            logger.info(f"Number of indexes to construct: {n_indexes}")
            
            # split all shards based on the GPU number
            # then merge the shards that belong to the same GPU to reduce GPU memory consumption
            # here we let the last shard (with less samples) be allocated to the GPU with more shards
            shard_path_splits = np.array_split(shard_paths[::-1], n_indexes)[::-1]
            shard_path_splits = [paths[::-1] for paths in shard_path_splits]
            # shard_path_splits = np.array_split(shard_paths, n_gpu)
            self.index = IndexShards()
            for index_id, paths in enumerate(shard_path_splits):
                gpu_id = index_id % n_gpu
                logger.info(f"GPU {gpu_id} will load {len(paths)} shards.")
                
                merged_index = TorchInnerProductIndex(device=f'cuda:{gpu_id}', use_fp16=config.retriever.use_fp16)
                for path in tqdm(paths, desc=f"Loading torch shard indexes for GPU {gpu_id}"):
                    embeddings = np.load(path).astype(np.float32)
                    # normalize
                    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
                    merged_index.add(embeddings)
                    del embeddings
                    gc.collect()
                    
                if merged_index.ntotal > 0:
                    self.index.add(merged_index)

        self.corpus_file_path = os.path.join(config.retriever.corpus_path,
                                             config.retriever.corpus_file)
        self.corpus = load_corpus(self.corpus_file_path)
        
        assert len(self.corpus) == self.index.ntotal, f"Corpus size {len(self.corpus)} does not match index size {self.index.ntotal}"
        
        # for debugging
        self.id2origin_id = None
        if 'origin_id' in self.corpus[0]:
            self.id2origin_id = {idx: int(item['origin_id']) for idx, item in enumerate(self.corpus)}

        self.top_k = config.retriever.top_k
        self.batch_size = config.retriever.retrieval_batch_size

    def get_retriever_tag(self):
        return self.retriever_tag

    def set_encoder(self, encoder): 
        self.encoder = encoder
        self.encoder.eval()
        
    def get_encoder(self): 
        return self.encoder

    def retrieve_with_text(self, queries, top_k=None):
        start_time = time.perf_counter()

        if isinstance(queries, str):
            queries = [queries]

        if top_k is None:
            top_k = self.top_k

        if 0==top_k:
            return RetrieverOutput(
                results=[[] for _ in range(len(queries))], 
                start_time=start_time,
                end_time=time.perf_counter()
            )
        
        if getattr(self, 'encoder', None) is None:
            raise ValueError("Encoder is not loaded. Please config `retriever.encoder_name_or_path` for text retrieval.")

        results = []
        # for start_idx in tqdm(range(0, len(queries), self.batch_size), desc='Retrieval process: '):
        for start_idx in range(0, len(queries), self.batch_size):
            query_batch = queries[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.multi_gpu_encode(query_batch, batch_size=self.batch_size, is_query=True)
            # normalize
            batch_emb /= np.linalg.norm(batch_emb, axis=1, keepdims=True)
            batch_scores, batch_chunk_ids = self.index.search(batch_emb, k=top_k)
            batch_scores = batch_scores.tolist()
            batch_chunk_ids = batch_chunk_ids.tolist()

            # load_chunks is not vectorized, but is a python list approach
            flat_chunk_ids = sum(batch_chunk_ids, [])
            batch_chunks = load_chunks(self.corpus, flat_chunk_ids)
            # chunk them back
            batch_chunks = [batch_chunks[i*top_k : (i+1)*top_k] for i in range(len(batch_chunk_ids))]

            # for debugging
            if self.id2origin_id is not None:
                batch_chunk_ids = [[self.id2origin_id[chunk_id] for chunk_id in chunk_ids] for chunk_ids in batch_chunk_ids]

            results.extend([
                [ChunkItem(chunk_id=str(chunk_id), text=chunk, score=score) 
                 for chunk_id, chunk, score in zip(chunk_ids, chunks, scores) if chunk_id != -1]
                for chunk_ids, chunks, scores in zip(batch_chunk_ids, batch_chunks, batch_scores)
            ])

            del batch_emb, batch_scores, batch_chunk_ids, query_batch, flat_chunk_ids, batch_chunks
            torch.cuda.empty_cache()
            
        
        return RetrieverOutput(
            results=results,
            start_time=start_time,
            end_time=time.perf_counter()
        )
    
    def retrieve_with_embeddings(self, query_embeddings, top_k=None):
        """ query_embeddings should be normalized already """
        start_time = time.perf_counter()
        
        if top_k is None:
            top_k = self.top_k

        if 0==top_k:
            return RetrieverOutput(
                results=[[] for _ in range(len(query_embeddings))],
                start_time=start_time,
                end_time=time.perf_counter()
            )

        results = []
        # for start_idx in tqdm(range(0, len(query_embeddings), self.batch_size), desc='Retrieval process: '):
        for start_idx in range(0, len(query_embeddings), self.batch_size):
            batch_emb = query_embeddings[start_idx:start_idx + self.batch_size]
            batch_scores, batch_chunk_ids = self.index.search(batch_emb, k=top_k)
            batch_scores = batch_scores.tolist()
            batch_chunk_ids = batch_chunk_ids.tolist()

            # load_chunks is not vectorized, but is a python list approach
            flat_chunk_ids = sum(batch_chunk_ids, [])
            batch_chunks = load_chunks(self.corpus, flat_chunk_ids)
            # chunk them back
            batch_chunks = [batch_chunks[i*top_k : (i+1)*top_k] for i in range(len(batch_chunk_ids))]

            # for debugging
            if self.id2origin_id is not None:
                batch_chunk_ids = [[self.id2origin_id[chunk_id] for chunk_id in chunk_ids] for chunk_ids in batch_chunk_ids]

            results.extend([
                [ChunkItem(chunk_id=str(chunk_id), text=chunk, score=score) 
                 for chunk_id, chunk, score in zip(chunk_ids, chunks, scores) if chunk_id != -1]
                for chunk_ids, chunks, scores in zip(batch_chunk_ids, batch_chunks, batch_scores)
            ])

            del batch_emb, batch_scores, batch_chunk_ids, flat_chunk_ids, batch_chunks
            torch.cuda.empty_cache()

        return RetrieverOutput(
            results=results,
            start_time=start_time,
            end_time=time.perf_counter()
        )
