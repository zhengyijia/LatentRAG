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

# from model.agentcpm import AgentCPM
from model.agentcpm_with_repetition_penalty import AgentCPM
from model.few_shot_rag import Few_Shot_RAG
from model.few_shot_rag_cot import Few_Shot_RAG_CoT
from model.ircot import IRCoT
from model.searchr1 import SearchR1
from model.r1searcher import R1Searcher
from model.zerosearch import ZeroSearch
from model.autorefine import AutoRefine
from model.infer_inputs_outputs import IterationRecord
from retriever.encoder import Encoder
# from retriever.faiss_retriever_gpu import FaissRetriever, EmptyRetriever
from retriever.torch_retriever_gpu import TorchRetriever, EmptyRetriever
from retriever.rest_retriever import RESTRetriever
from utils import calculate_em_score, calculate_cover_em_score, calculate_f1_score, calculate_search_success_score, flatten_list, unflatten_list

logger = logging.getLogger(__name__)

MODEL_CLASS_DICT = {
    'AgentCPM': AgentCPM,
    'Search-R1': SearchR1, 
    'IRCoT': IRCoT,
    'AutoRefine': AutoRefine,
    'R1Searcher': R1Searcher,
    'ZeroSearch': ZeroSearch,
    'Few-Shot-LLM': Few_Shot_RAG,
    'Few-Shot-RAG': Few_Shot_RAG, 
    'Few-Shot-CoT': Few_Shot_RAG_CoT, 
    'Few-Shot-RAG-CoT': Few_Shot_RAG_CoT
}

def batch_rollout(batch_data, retriever, model, config):
    top_k = config.retriever.top_k
    neg_k = config.retriever.get('neg_k', 0)

    finished = OrderedDict((query_id, None) for query_id in batch_data['id'])
    if config.model.model_class in ("AgentCPM", "Search-R1", "AutoRefine", "R1Searcher", "ZeroSearch"):
        # without initial retrieval
        inputs = [
            IterationRecord(
                query_id = query_id,
                query = question,
                golden_answers = [gold_answers] if isinstance(gold_answers, str) else gold_answers,
            )
            for query_id, question, gold_answers in zip(batch_data['id'], batch_data['question'], batch_data['golden_answers'])
        ]
    else:
        # with initial retrieval
        questions = [question.strip() for question in batch_data['question']]
        retriever_outputs = retriever.retrieve_with_text(questions, top_k=top_k + neg_k)
        inputs = [
            IterationRecord(
                query_id = query_id,
                query = question,
                golden_answers = [gold_answers] if isinstance(gold_answers, str) else gold_answers,
                chunk_items = [chunk_items[:top_k]],
                hard_neg_chunk_items = [chunk_items[top_k:]],
            )
            for query_id, question, gold_answers, chunk_items in zip(
                batch_data['id'], batch_data['question'], batch_data['golden_answers'], retriever_outputs.results
            )
        ]

    num_prompt_tokens = 0
    num_completion_tokens = 0
    while True: 
        outputs, metadata = model.batch_generate(inputs)
        num_prompt_tokens += metadata['prompt_tokens']
        num_completion_tokens += metadata['completion_tokens']
        logger.info(f"Processed {len(outputs)} queries, total prompt tokens: {num_prompt_tokens}, total completion tokens: {num_completion_tokens}")
        unfinished = []
        for i, item in enumerate(outputs):
            if item.answer is not None:
                finished[item.query_id] = deepcopy(item)
            else:
                unfinished.append(item)

        if len(unfinished) == 0:
            break
        
        # perform retrieval
        # all_subqueries = [item.subqueries[-1].strip() for item in unfinished]
        # retriever_outputs = retriever.retrieve_with_text(all_subqueries, top_k=top_k + neg_k)
        all_subqueries = [item.subqueries[-1] for item in unfinished]
        if isinstance(all_subqueries[0], str):
            all_subqueries = [subquery.strip() for subquery in all_subqueries]
            flat_all_subqueries = all_subqueries
        elif isinstance(all_subqueries[0], list):
            all_subqueries = [[subquery.strip() for subquery in subqueries] for subqueries in all_subqueries]
            all_num_subqueries = [len(subqueries) for subqueries in all_subqueries]
            flat_all_subqueries = [subquery for subqueries in all_subqueries for subquery in subqueries]
        
        retriever_outputs = retriever.retrieve_with_text(flat_all_subqueries, top_k=top_k + neg_k)
        # unflat results
        if isinstance(all_subqueries[0], list):
            retriever_outputs.results = unflatten_list(retriever_outputs.results, all_num_subqueries)
        
        multiple_subqueries = isinstance(all_subqueries[0], list)
        for result, item in zip(retriever_outputs.results, unfinished):
            if multiple_subqueries:
                item.chunk_items.append([sublist[:top_k] for sublist in result])
                item.hard_neg_chunk_items.append([sublist[top_k:] for sublist in result])
            else:
                item.chunk_items.append(result[:top_k])
                item.hard_neg_chunk_items.append(result[top_k:])

        inputs = unfinished
        
    if config.vllm.use_vllm:
        model.reset_prefix_cache()

    return finished.values(), {'prompt_tokens': num_prompt_tokens, 'completion_tokens': num_completion_tokens}


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    # for debug
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        logger.info(">>>>>>>>>> USING DEBUG MODE <<<<<<<<<<")
        
    OmegaConf.resolve(config)
    
    # print initial config
    logger.info(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    
    set_seed(config.training.seed)
    
    data_paths = []
    results_save_paths = []
    for eval_path in config.data.eval_files:
        data_path = os.path.join(
            config.data.data_path,
            eval_path
        )
        data_paths.append(data_path)

        param_tag = f'{config.model.model_tag}_{config.retriever.retriever_tag}'
        if getattr(config.model, 'max_turns', None):
            param_tag += f'_maxturns_{config.model.max_turns}'
        if getattr(config.retriever, 'top_k', None):
            param_tag += f'_topk_{config.retriever.top_k}'
        if getattr(config.retriever, 'neg_k', None):
            param_tag += f'_negk_{config.retriever.neg_k}'
        if getattr(config.data, 'exclude_chunk_text', False):
            param_tag += f'_exclude_chunk_text'
        save_filename = f'{param_tag}_reasoning_steps.json'
        # for debug
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            save_filename = f'DEBUG_{save_filename}'
        results_save_path = os.path.join(
            config.data.save_path,
            os.path.dirname(eval_path),
            os.path.splitext(os.path.basename(eval_path))[0],
            save_filename
        )
        results_save_paths.append(results_save_path)

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

    # load model
    model_cls = MODEL_CLASS_DICT.get(config.model.model_class, None)
    if model_cls is None:
        raise NotImplementedError(f"Model {config.model.model_class} is not implemented.")
    model = model_cls.load(config)

    for data_path, save_path in zip(data_paths, results_save_paths): 
        if os.path.exists(save_path) and not config.training.overwrite_output_dir:
            logger.warning(f"Results file {save_path} already exists, skip generation.")
            continue
        elif os.path.exists(save_path):
            os.remove(save_path)
            logger.warning(f"Cleaned up existing results file {save_path} before generation.")
            
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.basename(save_path)

        split = os.path.splitext(os.path.basename(data_path))[0]
        dataset = load_dataset("json", data_files={split: data_path})[split]
        
        results = []
        num_prompt_tokens = 0 
        num_completion_tokens = 0
        batch_size = config.data.batch_size
        batch_size = len(dataset) if batch_size == -1 else batch_size
        start_time = time.perf_counter()
        data_begin = getattr(config.data, 'data_range_begin', 0)
        data_begin = max(data_begin, 0)
        data_end = getattr(config.data, 'data_range_end', len(dataset)-1)
        data_end = min(data_end, len(dataset)-1)
        for start_idx in tqdm(range(data_begin, data_end+1, batch_size), desc=f"Processing data from {data_path}"):
            end_idx = min(start_idx + batch_size-1, data_end)
            logger.info(f"Processing batch data {start_idx}-{end_idx}")
            cache_file = os.path.splitext(save_file)[0] + f"_batch_{start_idx:04d}_{end_idx:04d}.json"
            cache_path = os.path.join(save_dir, cache_file)
            if getattr(config.data, 'cache_batch_results', False) and os.path.exists(cache_path):
                logger.info(f"Loading cached results from {cache_path} for batch {start_idx}-{end_idx}")
                with open(cache_path, 'r') as f:
                    batch_cache = json.load(f)
                results.extend(batch_cache['results'])
                num_prompt_tokens += batch_cache['num_prompt_tokens']
                num_completion_tokens += batch_cache['num_completion_tokens']
            else:
                batch_data = dataset[start_idx:end_idx+1]
                batch_results, metadata = batch_rollout(batch_data, retriever, model, config)
                
                # embeddings cannot be serialized
                for item in batch_results:
                    item.query_embedding = None
                    item.subquery_embeddings = None
                batch_results = [asdict(item) for item in batch_results]
                
                for item in batch_results:
                    golden_answers = item['golden_answers']
                    predicted_answer = item['answer']
                    em_score = calculate_em_score(golden_answers, predicted_answer)
                    cover_em_score = calculate_cover_em_score(golden_answers, predicted_answer)
                    f1_score = calculate_f1_score(golden_answers, predicted_answer)
                    item['em_score'] = em_score
                    item['cover_em_score'] = cover_em_score
                    item['f1_score'] = f1_score
                    search_success = calculate_search_success_score(golden_answers, [chunk_item['text'] for chunk_item in flatten_list(item['chunk_items'])])
                    item['search_success'] = search_success
                
                if getattr(config.data, 'exclude_chunk_text', False):
                    for item in batch_results:
                        for chunk_items in item['chunk_items']:
                            for chunk_item in chunk_items:
                                chunk_item['text'] = None
                        for hard_neg_chunk_items in item['hard_neg_chunk_items']:
                            for chunk_item in hard_neg_chunk_items:
                                chunk_item['text'] = None
                
                results.extend(batch_results)
                num_prompt_tokens += metadata['prompt_tokens']
                num_completion_tokens += metadata['completion_tokens']
                
                if getattr(config.data, 'cache_batch_results', False):
                    batch_cache = {
                        'results': batch_results,
                        'num_prompt_tokens': metadata['prompt_tokens'],
                        'num_completion_tokens': metadata['completion_tokens']
                    }
                    with open(cache_path, 'w') as f:
                        json.dump(batch_cache, f)
                    logger.info(f"Saved cached results to {cache_path} for batch {start_idx}-{end_idx}")
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Processing time for {data_path}: {total_time:.4f} seconds")
        
        all_em_scores = []
        all_cover_em_scores = []
        all_f1_scores = []
        all_search_success_scores = []
        for i, item in enumerate(results):
            all_em_scores.append(item['em_score'])
            all_cover_em_scores.append(item['cover_em_score'])
            all_f1_scores.append(item['f1_score'])
            all_search_success_scores.append(item['search_success'])
        results = {
            "model_name_or_path": config.model.model_name_or_path,
            "encoder_name_or_path": config.retriever.encoder_name_or_path,
            'avg_em_score': np.mean(all_em_scores),
            'avg_cover_em_score': np.mean(all_cover_em_scores),
            'avg_f1_score': np.mean(all_f1_scores),
            'avg_search_success_score': np.mean(all_search_success_scores),
            'num_samples': len(results),
            'num_prompt_tokens': num_prompt_tokens,
            'num_completion_tokens': num_completion_tokens,
            'total_time': total_time,
            'batch_size': batch_size,
            'results': results
        }
        logger.info(f"Input tokens: {num_prompt_tokens}")
        logger.info(f"Output tokens: {num_completion_tokens}")
        logger.info(f"Average EM score: {results['avg_em_score']:.4f}")
        logger.info(f"Average Cover EM score: {results['avg_cover_em_score']:.4f}")
        logger.info(f"Average F1 score: {results['avg_f1_score']:.4f}")
        logger.info(f"Average Search Success score: {results['avg_search_success_score']:.4f}")

        # add data range to the file name if specified
        additional_param_tag = ''
        if getattr(config.data, 'data_range_begin', None) is not None:
            additional_param_tag += f'_begin_{config.data.data_range_begin}'
        if getattr(config.data, 'data_range_end', None) is not None:
            additional_param_tag += f'_end_{config.data.data_range_end}'
            
        save_file = os.path.splitext(save_file)[0] + f"{additional_param_tag}.json"
        save_path = os.path.join(save_dir, save_file)

        # save results
        with open(save_path, 'w') as f:
            json.dump(results, f)
        logger.info(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
