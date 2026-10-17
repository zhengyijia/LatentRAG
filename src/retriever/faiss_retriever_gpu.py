# modified from Search-R1
# https://github.com/PeterGriffinJin/Search-R1/blob/e23b87911693e5b174fc4d2d6bfd21f99137b5fc/search_r1/search/retrieval_server.py

import os
import glob
import re
import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import torch
import logging
import time
import jsonlines
import gc

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

class FaissRetriever(): 

    def __init__(self, config):
        self.retriever_tag = config.retriever.retriever_tag
        
        # if config.retriever.encoder_name_or_path is not None:
        #     self.encoder = Encoder.load(config)
        #     self.encoder.eval()

        self.corpus_index_folder = os.path.join(config.retriever.corpus_path,
                                                config.retriever.corpus_index_folder)
        logger.info(f"Loading FAISS index from {self.corpus_index_folder}")
        all_paths = glob.glob(os.path.join(self.corpus_index_folder, "shard_index_*.npy"))
        shard_paths = [p for p in all_paths if re.search(r"shard_index_\d{4}\.npy$", p)]
        assert len(shard_paths) > 0
        
        def extract_number(path):
            match = re.search(r"shard_index_(\d{4})\.npy$", path)
            return int(match.group(1)) if match else -1
        shard_paths.sort(key=extract_number)

        if not config.retriever.faiss_gpu:
            shard_indexes = []
            for path in tqdm(shard_paths, desc="Loading FAISS shard indexes"):
                embeddings = np.load(path).astype(np.float32)
                # normalize
                embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
                if config.retriever.use_fp16:
                    shard_index = faiss.index_factory(embeddings.shape[1], "SQfp16", faiss.METRIC_INNER_PRODUCT)
                else:
                    shard_index = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
                shard_index.add(embeddings)
                shard_indexes.append(shard_index)
                
                del embeddings
                gc.collect()

            self.index = faiss.IndexShards(shard_indexes[0].d, True, True)
            for shard_index in shard_indexes:
                self.index.add_shard(shard_index)
        else:
            n_gpu = faiss.get_num_gpus()
            logger.info(f"Number of available GPUs: {n_gpu}")
            
            # gpu_resources = [faiss.StandardGpuResources() for _ in range(n_gpu)] 
            # for gpu_resource in gpu_resources:
            #     gpu_resource.noTempMemory()
            
            # num index
            n_indexes = n_gpu * 2  # Set to larger than 1 to avoid OOM when constructing index on GPU
            logger.info(f"Number of indexes to construct: {n_indexes}")
            
            # split all shards based on the GPU number
            # then merge the shards that belong to the same GPU to reduce GPU memory consumption
            # here we let the last shard (with less samples) be allocated to the GPU with more shards
            shard_path_splits = np.array_split(shard_paths[::-1], n_indexes)[::-1]
            shard_path_splits = [paths[::-1] for paths in shard_path_splits]
            # shard_path_splits = np.array_split(shard_paths, n_gpu)
            shard_indexes = []
            for index_id, paths in enumerate(shard_path_splits):
                gpu_id = index_id % n_gpu
                logger.info(f"GPU {gpu_id} will load {len(paths)} shards.")
                merged_index_cpu = None
                for path in tqdm(paths, desc=f"Loading FAISS shard indexes for GPU {gpu_id}"):
                    embeddings = np.load(path).astype(np.float32)
                    # normalize
                    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
                    if config.retriever.use_fp16:
                        shard_index_cpu = faiss.index_factory(embeddings.shape[1], "SQfp16", faiss.METRIC_INNER_PRODUCT)
                    else:
                        shard_index_cpu = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
                    shard_index_cpu.add(embeddings)
                    del embeddings
                    gc.collect()
                    
                    if merged_index_cpu is None:
                        merged_index_cpu = shard_index_cpu
                    else:
                        merged_index_cpu.merge_from(shard_index_cpu, 0)
                        shard_index_cpu.reset()
                        del shard_index_cpu
                        gc.collect()
                        
                if merged_index_cpu is not None:
                    gpu_resource = faiss.StandardGpuResources()
                    gpu_resource.noTempMemory()
                    # gpu_resource.setTempMemory(512 * 1024 * 1024)
                    # gpu_resource.setPinnedMemory(512 * 1024 * 1024)
                    # gpu_resource = gpu_resources[gpu_id]
                    gpu_options = faiss.GpuClonerOptions()
                    if config.retriever.use_fp16:
                        gpu_options.useFloat16 = True
                    shard_index_gpu = faiss.index_cpu_to_gpu(gpu_resource, gpu_id, merged_index_cpu, gpu_options)
                    shard_indexes.append(shard_index_gpu)
                    merged_index_cpu.reset()
                    del merged_index_cpu
                    gc.collect()
                
            self.index = faiss.IndexShards(shard_indexes[0].d, True, True)
            for shard_index in shard_indexes:
                self.index.add_shard(shard_index)

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
