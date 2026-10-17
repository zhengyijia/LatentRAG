import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict
from typing import List, Dict, Optional, Union
import numpy as np

import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from retriever.encoder import Encoder
# from retriever.faiss_retriever_gpu import FaissRetriever, EmptyRetriever
from retriever.torch_retriever_gpu import TorchRetriever, EmptyRetriever

logger = logging.getLogger(__name__)

app = FastAPI()
retriever = None

class QueryRequest(BaseModel):
    queries: Union[List[str], List[List[float]]]
    topk: Optional[int] = None
    mode: Optional[str] = 'text'

@app.get("/retriever_info")
def get_retriever_info():
    global retriever
    return {
        "retriever_tag": retriever.get_retriever_tag(),
    }

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    global retriever
    
    if request.mode == 'text':
        retriever_outputs = retriever.retrieve_with_text(request.queries, top_k=request.topk)
    elif request.mode == 'embedding':
        request.queries = np.array(request.queries, dtype=np.float32)
        retriever_outputs = retriever.retrieve_with_embeddings(request.queries, top_k=request.topk)
    else:
        return {'result': None, 'success': False, 'error': f"Unsupported retrieval mode: {request.mode}"}
        
    # Format response
    resp = []
    results = retriever_outputs.results
    for i, single_result in enumerate(results):
        combined = []
        for item in single_result:
            combined.append({"document": item.text, "score": item.score, "chunk_id": item.chunk_id})
        resp.append(combined)
        
    return {'result': resp, 'success': True}

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    global retriever
    
    # # wrap retriever config
    # config = OmegaConf.create({"retriever": config})
    
    # build retriever
    if config.retriever.retriever_tag == 'none':
        retriever = EmptyRetriever(config)
    else:
        retriever = TorchRetriever(config)
    
    reference_encoder = Encoder.load(config, trainable=False, use_lora=False)
    retriever.set_encoder(reference_encoder)
    
    # for debug
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        logger.info(">>>>>>>>>> USING DEBUG MODE <<<<<<<<<<")
        test()
    
    uvicorn.run(app, host=config.server.host, port=config.server.port)

def test():
    client = TestClient(app)
    
    response = client.get("/retriever_info")
    print(response.json())
    
    data = {
        "queries": ["What is Latent Search?", "What is retriever model?"],
        "topk": 3,
        "return_scores": True, # ignored
        "mode": "text"
    }
    response = client.post("/retrieve", json=data)
    print(response.json())
    
    embedding_size = retriever.encoder.get_embedding_size()
    embedding_data = np.random.rand(2, embedding_size).tolist()
    data = {
        "queries": embedding_data,
        "topk": 3,
        "return_scores": True, # ignored
        "mode": "embedding"
    }
    response = client.post("/retrieve", json=data)
    print(response.json())

if __name__ == "__main__":
    main()
    
