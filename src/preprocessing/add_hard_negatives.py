import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset
from copy import deepcopy
from dataclasses import asdict
import time

from model.infer_inputs_outputs import IterationRecord
from retriever.encoder import Encoder
# from retriever.faiss_retriever_gpu import FaissRetriever, EmptyRetriever
from retriever.torch_retriever_gpu import TorchRetriever, EmptyRetriever
from retriever.rest_retriever import RESTRetriever
from utils import calculate_em_score, calculate_cover_em_score, calculate_f1_score, calculate_search_success_score, flatten_list, unflatten_list

logger = logging.getLogger(__name__)

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    # for debug
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        logger.info(">>>>>>>>>> USING DEBUG MODE <<<<<<<<<<")
        
    OmegaConf.resolve(config)

    # build retriever
    if config.retriever.retriever_tag == 'none':
        retriever = EmptyRetriever(config)
    elif getattr(config.retriever, 'base_url', None) is not None:
        logger.info(f"Using REST retriever with base url: {config.retriever.base_url}")
        retriever = RESTRetriever(config)
    else:
        retriever = TorchRetriever(config)

    reference_encoder = Encoder.load(config, trainable=False, use_lora=False)
    reference_encoder.eval()
    retriever.set_encoder(reference_encoder)

    data_paths = []
    results_save_paths = []
    for eval_path in config.data.eval_files:
        data_path = os.path.join(
            config.data.data_path,
            eval_path
        )
        data_paths.append(data_path)

        save_path = os.path.dirname(data_path)
        neg_k = config.retriever.neg_k
        save_filename = os.path.splitext(os.path.basename(data_path))[0] + f"_hard_negatives_{neg_k}.jsonl"
        save_path = os.path.join(save_path, save_filename)

        results_save_paths.append(save_path)

    top_k = config.retriever.top_k
    neg_k = config.retriever.neg_k
    k = top_k + neg_k
    for data_path, save_path in zip(data_paths, results_save_paths):
        logger.info(f"Adding hard negatives for data file: {data_path}")
        logger.info(f"Results will be saved to: {save_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # add hard negatives
        for record in tqdm(data['results'], desc="Adding hard negatives"):
            subqueries = record['subqueries']

            retriever_outputs = retriever.retrieve_with_text(subqueries, top_k=k).results
            retriever_outputs = [sublist[top_k:] for sublist in retriever_outputs]
            retriever_outputs = [[asdict(item) for item in sublist] for sublist in retriever_outputs]
            record['hard_neg_chunk_items'] = retriever_outputs

        # save new data with hard negatives
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Finished adding hard negatives for data file: {data_path}")

if __name__ == "__main__":
    main()
