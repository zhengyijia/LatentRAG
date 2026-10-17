# modified from Search-R1
# https://github.com/PeterGriffinJin/Search-R1/blob/e23b87911693e5b174fc4d2d6bfd21f99137b5fc/search_r1/search/retrieval_server.py

import os
import glob
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import logging
import time
import requests
# from pydantic import BaseModel

from retriever.encoder import Encoder

logger = logging.getLogger(__name__)

@dataclass
class ChunkItem:
    chunk_id: str
    text: str
    score: float
    
    @classmethod
    def from_dict(cls, d):
        return cls(
            chunk_id=d["chunk_id"],
            text=d["text"],
            score=d["score"]
        )

@dataclass
class RetrieverOutput:
    results: List[List[ChunkItem]]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_start_time: Optional[float] = None
    total_end_time: Optional[float] = None

class RESTRetriever(): 

    def __init__(self, config):
        self.retriever_tag = config.retriever.retriever_tag
        self.base_url = config.retriever.base_url
        response = requests.get(f"{self.base_url}/retriever_info")
        assert response.status_code == 200, f"Failed to get retriever info from REST retriever server: {response.text}"
        response = response.json()
        assert response['retriever_tag'] == self.retriever_tag, f"Retriever tag mismatch: expected {self.retriever_tag}, got {response['retriever_tag']}"

    def get_retriever_tag(self):
        return self.retriever_tag

    def set_encoder(self, encoder): 
        self.encoder = encoder
        logger.warning("`set_encoder` does not change the encoder used by the REST retriever server!")
        
    def get_encoder(self): 
        return self.encoder

    def retrieve_with_text(self, queries, top_k=None):
        total_start_time = time.perf_counter()
        
        data = {
            "queries": queries,
            "topk": top_k,
            "mode": "text"
        }
        response = requests.post(f"{self.base_url}/retrieve", json=data)
        assert response.status_code == 200, f"Failed to retrieve from REST retriever server: {response.text}"
        
        response = response.json().get('results')
        
        results = [
            [ChunkItem.from_dict(item) for item in result]
            for result in response.get('results')
        ]
        start_time = response.get('start_time')
        end_time = response.get('end_time')
        
        return RetrieverOutput(
            results=results,
            start_time=start_time,
            end_time=end_time,
            total_start_time=total_start_time,
            total_end_time=time.perf_counter()
        )
    
    def retrieve_with_embeddings(self, query_embeddings, top_k=None):
        """ query_embeddings should be normalized already """
        total_start_time = time.perf_counter()

        query_embeddings = query_embeddings.tolist()
        data = {
            "queries": query_embeddings,
            "topk": top_k,
            "mode": "embedding"
        }
        response = requests.post(f"{self.base_url}/retrieve", json=data)
        assert response.status_code == 200, f"Failed to retrieve from REST retriever server: {response.text}"
        
        response = response.json().get('results')
        
        results = [
            [ChunkItem.from_dict(item) for item in result]
            for result in response.get('results')
        ]
        start_time = response.get('start_time')
        end_time = response.get('end_time')

        return RetrieverOutput(
            results=results,
            start_time=start_time,
            end_time=end_time,
            total_start_time=total_start_time,
            total_end_time=time.perf_counter()
        )
        
