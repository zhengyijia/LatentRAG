import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset
from copy import deepcopy
from dataclasses import asdict
import time
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from model.latentrag import LatentRAG
from model.infer_inputs_outputs import IterationRecord
from retriever.encoder import Encoder
from utils import calculate_em_score, calculate_f1_score, calculate_search_success_score, flatten_list

logger = logging.getLogger(__name__)

MODEL_CLASS_DICT = {
    'LatentRAG': LatentRAG,
}


def rollout(data, model, config):
    inputs = IterationRecord(
        query_id = data['query_id'],
        query = data['query'],
        golden_answers = [data['golden_answers']] if isinstance(data['golden_answers'], str) else data['golden_answers'],
        reference_chunk_items = data['chunk_items'],
        reference_subqueries = data['subqueries'],
        reference_thoughts = data['thoughts'],
        reference_subquery_embeddings = None,
        reference_em_score = data['em_score'],
        reference_answer = data['answer'],
        reference_search_success = data['search_success'],
        reference_force_answer = data['force_answer'],
    )
    
    if config.model.model_class in ("LatentRAG"):
        latent_thought_logits_info, latent_subquery_logits_info = model.get_latent_logits(inputs)
    else:
        raise NotImplementedError(f"Model class {config.model.model_class} is not implemented for teacher forcing evaluation.")
    
    return latent_thought_logits_info, latent_subquery_logits_info

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):

    # for debug
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        logger.info(">>>>>>>>>> USING DEBUG MODE <<<<<<<<<<")
    
    # print initial config
    logger.info(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    set_seed(config.training.seed)

    data_paths = []
    results_save_paths = []
    for eval_filepath in config.data.eval_files:
        data_path = os.path.join(
            config.data.data_path,
            eval_filepath
        )
        data_paths.append(data_path)

        file_prefix = 'LATENT_LOGITS'
        if config.data.get('sampling', -1) > 0:
            file_prefix += f"_SAMPLED_{config.data.sampling}"
        if config.data.get('filter_em_correct', False):
            file_prefix += f"_FILTER_EM_CORRECT"
        if config.data.get('filter_em_incorrect', False):
            file_prefix += f"_FILTER_EM_INCORRECT"
        save_filename = os.path.basename(eval_filepath)
        save_filename = file_prefix + '_' + save_filename
        # for debug
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            save_filename = f'DEBUG_{save_filename}'
        save_filename = os.path.join(os.path.dirname(eval_filepath), save_filename)

        results_save_path = os.path.join(
            config.data.save_path,
            save_filename
        ) if config.data.save_path is not None else None
        results_save_paths.append(results_save_path)

    # load model
    model_cls = MODEL_CLASS_DICT.get(config.model.model_class, None)
    if model_cls is None:
        raise NotImplementedError(f"Model {config.model.model_class} is not implemented.")
    model = model_cls.load(config)
    if config.model.model_class in ('LatentRAG'):
        encoder_use_lora = config.retriever.get("encoder_lora", {}).get("use_lora", False)
        encoder = Encoder.load(config, trainable=False, use_lora=encoder_use_lora, custom_encoder=True)
        model.set_encoder(encoder)
    model.eval()
    
    for data_path, save_path in zip(data_paths, results_save_paths): 
        if save_path is not None: 
            if os.path.exists(save_path) and not config.training.overwrite_output_dir:
                logger.warning(f"Results file {save_path} already exists, skip generation.")
                continue
            elif os.path.exists(save_path):
                os.remove(save_path)

        # split = os.path.splitext(os.path.basename(data_path))[0]
        split = 'none'
        dataset = load_dataset("json", data_files={split: data_path}, field="results")[split]
        logger.info(f"Loaded {len(dataset)} samples from {data_path}")
        if config.data.get('filter_em_correct', False):
            dataset = dataset.filter(lambda x: x["em_score"] > 0)
            logger.info(f"Filtered dataset to {len(dataset)} samples with EM score > 0")
        if config.data.get('filter_em_incorrect', False):
            dataset = dataset.filter(lambda x: x["em_score"] == 0)
            logger.info(f"Filtered dataset to {len(dataset)} samples with EM score == 0")
        if config.data.get('sampling', -1) > 0:
            sampling = min(config.data.sampling, len(dataset))
            dataset = dataset.shuffle(seed=42).select(range(sampling))
            logger.info(f"Using {sampling} samples from {data_path}")

        results = []
        start_time = time.perf_counter()
        for idx in tqdm(range(0, len(dataset)), desc=f"Processing data from {data_path}"):
            data = dataset[idx]
            latent_thought_logits_info, latent_subquery_logits_info = rollout(data, model, config)
            data['latent_thought_logits_info'] = latent_thought_logits_info
            data['latent_subquery_logits_info'] = latent_subquery_logits_info
            results.append(data)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_query = total_time / len(dataset) if len(dataset) > 0 else 0
        logger.info(f"Processing time for {data_path}: {total_time:.4f} seconds")
        logger.info(f"Average time per query for {data_path}: {avg_time_per_query:.4f} seconds")
        
        # save results
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            # with open(save_path, 'w') as f:
            #     json.dump(results, f)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()


