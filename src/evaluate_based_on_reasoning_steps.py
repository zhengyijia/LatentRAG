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

use_wandb = os.getenv("USE_WANDB", "true").lower() == "true"
if use_wandb:
    import wandb
    api = wandb.Api()

from model.searchr1 import SearchR1
from model.latentrag import LatentRAG
from model.sft_model import SFTModel
from model.infer_inputs_outputs import IterationRecord
from retriever.encoder import Encoder
# from retriever.faiss_retriever_gpu import FaissRetriever, EmptyRetriever
from retriever.torch_retriever_gpu import TorchRetriever, EmptyRetriever
from retriever.rest_retriever import RESTRetriever
from utils import calculate_em_score, calculate_cover_em_score, calculate_f1_score, calculate_search_success_score, flatten_list, unflatten_list

logger = logging.getLogger(__name__)

MODEL_CLASS_DICT = {
    'Search-R1': SearchR1, 
    'LatentRAG': LatentRAG,
    'SFTModel': SFTModel,
}

def batch_rollout(batch_data, retriever, model, config):
    top_k = config.retriever.top_k
    
    batch_data['subqueries'] = [
        [subquery for subquery in sublist if subquery is not None]
        for sublist in batch_data['subqueries']
    ]
    finished = OrderedDict((query_id, None) for query_id in batch_data['query_id'])
    
    reference_llm_judge_results = batch_data['llm_judge_result'] if 'llm_judge_result' in batch_data else [None for _ in batch_data['query_id']]
    reference_cover_em_scores = batch_data['cover_em_score'] if 'cover_em_score' in batch_data else [None for _ in batch_data['query_id']]
    if config.model.model_class in ("Search-R1", "LatentRAG", "SFTModel"):
        # without initial retrieval
        inputs = [
            IterationRecord(
                query_id = query_id,
                query = question,
                golden_answers = [gold_answers] if isinstance(gold_answers, str) else gold_answers,
                reference_chunk_items = reference_chunk_items, 
                reference_subqueries = reference_subqueries, 
                reference_thoughts = reference_thoughts,
                reference_subquery_embeddings = retriever.encoder.single_batch_encode(reference_subqueries) if len(reference_subqueries) > 0 else None,
                first_reference_topk_chunk_embeddings = retriever.encoder.single_batch_encode([chunk_item['text'] for chunk_item in reference_chunk_items[0]]) if len(reference_chunk_items) > 0 else None,
                reference_em_score = reference_em_score,
                reference_cover_em_score = reference_cover_em_score,
                reference_answer = reference_answer,
                reference_search_success = reference_search_success,
                reference_llm_judge_result = reference_llm_judge_result,
            )
            for query_id, question, gold_answers, reference_chunk_items, reference_subqueries, reference_thoughts, reference_em_score, reference_cover_em_score, reference_answer, reference_search_success, reference_llm_judge_result in 
            zip(batch_data['query_id'], batch_data['query'], 
                batch_data['golden_answers'], batch_data['chunk_items'], 
                batch_data['subqueries'], 
                batch_data['thoughts'],
                batch_data['em_score'], reference_cover_em_scores, batch_data['answer'], batch_data['search_success'], 
                reference_llm_judge_results)
        ]
    else:
        # with initial retrieval
        questions = [question.strip() for question in batch_data['query']]
        retriever_outputs = retriever.retrieve_with_text(questions, top_k=top_k)
        inputs = [
            IterationRecord(
                query_id = query_id,
                query = question,
                golden_answers = [gold_answers] if isinstance(gold_answers, str) else gold_answers,
                reference_chunk_items = reference_chunk_items, 
                reference_subqueries = reference_subqueries, 
                reference_thoughts = reference_thoughts,
                reference_em_score = reference_em_score,
                reference_cover_em_score = reference_cover_em_score,
                reference_answer = reference_answer,
                reference_search_success = reference_search_success,
                chunk_items = [chunk_items],
            )
            for query_id, question, gold_answers, reference_chunk_items, reference_subqueries, reference_thoughts, reference_em_score, reference_cover_em_score, reference_answer, reference_search_success, chunk_items in zip(
                batch_data['query_id'], batch_data['query'], 
                batch_data['golden_answers'], batch_data['chunk_items'],
                batch_data['subqueries'], 
                batch_data['thoughts'],
                batch_data['em_score'], reference_cover_em_scores, batch_data['answer'], batch_data['search_success'], 
                retriever_outputs.results, 
            )
        ]

    num_prompt_tokens = 0
    num_completion_tokens = 0
    while True: 
        kwargs = {}
        if config.get('batch_subquery_decoding', False):
            kwargs['batch_subquery_decoding'] = True
        
        outputs, metadata = model.batch_generate(inputs, **kwargs)
        
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
        if len(unfinished[0].subquery_embeddings)== 0 or unfinished[0].subquery_embeddings[-1] is None: 
            all_subqueries = [item.subqueries[-1] for item in unfinished]
            if isinstance(all_subqueries[0], str):
                all_subqueries = [subquery.strip() for subquery in all_subqueries]
                flat_all_subqueries = all_subqueries
            elif isinstance(all_subqueries[0], list):
                all_subqueries = [[subquery.strip() for subquery in subqueries] for subqueries in all_subqueries]
                all_num_subqueries = [len(subqueries) for subqueries in all_subqueries]
                flat_all_subqueries = [subquery for subqueries in all_subqueries for subquery in subqueries]
            
            retriever_outputs = retriever.retrieve_with_text(flat_all_subqueries, top_k=top_k)
            
            # unflat results
            if isinstance(all_subqueries[0], list):
                retriever_outputs.results = unflatten_list(retriever_outputs.results, all_num_subqueries)
        else: 
            all_subquery_embeddings = np.concatenate([item.subquery_embeddings[-1] for item in unfinished], axis=0)
            retriever_outputs = retriever.retrieve_with_embeddings(all_subquery_embeddings, top_k=top_k)
        
        for result, item in zip(retriever_outputs.results, unfinished):
            item.chunk_items.append(result)
            
        inputs = unfinished

    return list(finished.values()), {'prompt_tokens': num_prompt_tokens, 'completion_tokens': num_completion_tokens}


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

        file_prefix = 'EVAL'
        if config.data.get('sampling', -1) > 0:
            file_prefix += f"_sampled_{config.data.sampling}"
        if config.get('batch_subquery_decoding', False):
            file_prefix += f"_batch_subquery_decoding"
        batch_size = config.data.batch_size
        file_prefix += f"_bs{batch_size}"
        if config.vllm.get('use_vllm', False):
            file_prefix += f"_use_vllm"
        if config.data.get('filter_em_correct', False):
            file_prefix += f"_filter_em_correct"
        if config.data.get('filter_em_incorrect', False):
            file_prefix += f"_filter_em_incorrect"
        save_filename = file_prefix + '_' + eval_filepath.replace('/', '_')
        # for debug
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            save_filename = f'DEBUG_{save_filename}'

        results_save_path = os.path.join(
            config.data.save_path,
            save_filename
        ) if config.data.save_path is not None else None
        results_save_paths.append(results_save_path)

    # build retriever
    if config.retriever.retriever_tag == 'none':
        retriever = EmptyRetriever(config)
    elif getattr(config.retriever, 'base_url', None) is not None:
        logger.info(f"Using REST retriever with base url: {config.retriever.base_url}")
        retriever = RESTRetriever(config)
    else:
        retriever = TorchRetriever(config)

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
    
    # set reference encoder for retriever
    if config.retriever.retriever_tag != 'none':
        if config.model.model_class in ('LatentRAG') and config.retriever.get("share_encoder", False):
            reference_encoder = encoder
        else:
            reference_encoder = Encoder.load(config, trainable=False, use_lora=False)
        retriever.set_encoder(reference_encoder)

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
        num_prompt_tokens = 0
        num_completion_tokens = 0
        batch_size = config.data.batch_size
        batch_size = len(dataset) if batch_size == -1 else batch_size
        # batch_size = 1  # fix batch_size to 1 for recording request latency per sample
        start_time = time.perf_counter()
        for start_idx in tqdm(range(0, len(dataset), batch_size), desc=f"Processing data from {data_path}"):
            batch_data = dataset[start_idx:start_idx + batch_size]
            batch_results, metadata = batch_rollout(batch_data, retriever, model, config)
            results.extend(batch_results)
            num_prompt_tokens += metadata['prompt_tokens']
            num_completion_tokens += metadata['completion_tokens']
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_query = total_time / len(dataset) if len(dataset) > 0 else 0
        logger.info(f"Processing time for {data_path}: {total_time:.4f} seconds")
        logger.info(f"Average time per query for {data_path}: {avg_time_per_query:.4f} seconds")
        
        avg_infer_time_per_query = np.mean([np.sum(item.infer_time_each_steps) for item in results])
        avg_infer_time_subquery_steps = np.mean([np.sum(item.infer_time_each_steps[:-1]) for item in results if len(item.infer_time_each_steps) > 1])
        avg_infer_time_answer_steps = np.mean([item.infer_time_each_steps[-1] for item in results])
        logger.info(f"Average inference time per query for {data_path}: {avg_infer_time_per_query:.4f} seconds")
        logger.info(f"Average inference time for subquery steps for {data_path}: {avg_infer_time_subquery_steps:.4f} seconds")
        logger.info(f"Average inference time for answer step for {data_path}: {avg_infer_time_answer_steps:.4f} seconds")
        
        # embeddings cannot be serialized
        all_query_embeddings = [item.query_embedding for item in results if item.query_embedding is not None]
        all_subquery_embeddings = [np.concatenate(item.subquery_embeddings, axis=0)
                                   if len([emb for emb in item.subquery_embeddings if emb is not None]) > 0 else None
                                   for item in results]
        all_reference_subquery_embeddings = [item.reference_subquery_embeddings for item in results]
        all_first_reference_topk_chunk_embeddings = [item.first_reference_topk_chunk_embeddings for item in results]
        for item in results:
            item.query_embedding = None
            item.subquery_embeddings = None
            item.reference_subquery_embeddings = None
            item.first_reference_topk_chunk_embeddings = None
        results = [asdict(item) for item in results]
        
        # compute metrics
        all_recalls = []
        all_em_scores = []
        all_cover_em_scores = []
        all_f1_scores = []
        all_search_success_scores = []
        all_first_recall = []
        all_first_subquery_cosine = []
        all_first_top1_chunk_cosine = []
        all_first_topk_chunk_cosine = []
        all_first_max_pos_chunk_cosine = []
        confusion_matrix = [[0, 0], [0, 0]] # comparison with Search-R1, columns: Search-R1 correct/incorrect, rows: current model correct/incorrect
        for i, item in enumerate(results):
            first_reference_chunk_ids = set([chunk_item['chunk_id'] for chunk_item in item['reference_chunk_items'][0]]) if len(item['reference_chunk_items']) > 0 else set()
            first_chunk_ids = set([chunk_item['chunk_id'] for chunk_item in item['chunk_items'][0]]) if len(item['chunk_items']) > 0 else set()
            first_recall = len(first_reference_chunk_ids & first_chunk_ids) / len(first_reference_chunk_ids) if len(first_reference_chunk_ids) > 0 else 0.0
            
            reference_chunk_ids = set([chunk_item['chunk_id'] for chunk_items in item['reference_chunk_items'] for chunk_item in chunk_items])
            chunk_ids = set([chunk_item['chunk_id'] for chunk_items in item['chunk_items'] for chunk_item in chunk_items])
            recall = len(reference_chunk_ids & chunk_ids) / len(reference_chunk_ids) if len(reference_chunk_ids) > 0 else 0.0

            golden_answers = item['golden_answers']
            predicted_answer = item['answer']
            em_score = calculate_em_score(golden_answers, predicted_answer)
            cover_em_score = calculate_cover_em_score(golden_answers, predicted_answer)
            f1_score = calculate_f1_score(golden_answers, predicted_answer)
            search_success = calculate_search_success_score(golden_answers, [chunk_item['text'] for chunk_item in flatten_list(item['chunk_items'])])

            if all_subquery_embeddings[i] is not None and all_reference_subquery_embeddings[i] is not None:
                subquery_embedding = all_subquery_embeddings[i][0]
                reference_subquery_embedding = all_reference_subquery_embeddings[i][0]
                # normalize
                reference_subquery_embedding = reference_subquery_embedding / np.linalg.norm(reference_subquery_embedding)
                first_subquery_cosine = np.dot(subquery_embedding, reference_subquery_embedding)
                first_subquery_cosine = float(first_subquery_cosine)
            else:
                first_subquery_cosine = 0.0
                
            if all_subquery_embeddings[i] is not None and all_first_reference_topk_chunk_embeddings[i] is not None and len(all_first_reference_topk_chunk_embeddings[i]) > 0:
                subquery_embedding = all_subquery_embeddings[i][0]
                first_reference_topk_chunk_embeddings = all_first_reference_topk_chunk_embeddings[i]
                # normalize
                first_reference_topk_chunk_embeddings /= np.linalg.norm(first_reference_topk_chunk_embeddings, axis=1, keepdims=True)
                first_topk_chunk_cosine_scores = np.dot(first_reference_topk_chunk_embeddings, subquery_embedding)
                first_top1_chunk_cosine = float(first_topk_chunk_cosine_scores[0])
                first_topk_chunk_cosine = float(first_topk_chunk_cosine_scores.mean())
                first_max_pos_chunk_cosine = float(first_topk_chunk_cosine_scores.max())
            else:
                first_top1_chunk_cosine = 0.0
                first_topk_chunk_cosine = 0.0
                first_max_pos_chunk_cosine = 0.0

            item['recall'] = recall
            item['em_score'] = em_score
            item['cover_em_score'] = cover_em_score
            item['f1_score'] = f1_score
            item['search_success'] = search_success
            item['first_recall'] = first_recall
            item['first_subquery_cosine'] = first_subquery_cosine
            item['first_top1_chunk_cosine'] = first_top1_chunk_cosine
            item['first_topk_chunk_cosine'] = first_topk_chunk_cosine
            item['first_max_pos_chunk_cosine'] = first_max_pos_chunk_cosine
            
            all_recalls.append(recall)
            all_em_scores.append(em_score)
            all_cover_em_scores.append(cover_em_score)
            all_f1_scores.append(f1_score)
            all_search_success_scores.append(search_success)
            all_first_recall.append(first_recall)
            all_first_subquery_cosine.append(first_subquery_cosine)
            all_first_top1_chunk_cosine.append(first_top1_chunk_cosine)
            all_first_topk_chunk_cosine.append(first_topk_chunk_cosine)
            all_first_max_pos_chunk_cosine.append(first_max_pos_chunk_cosine)
            
            # confusion matrix
            reference_correct = 1 if item['reference_em_score'] > 0 else 0
            current_model_correct = 1 if em_score > 0 else 0
            confusion_matrix[1-current_model_correct][1-reference_correct] += 1
            item['comparison_with_reference'] = f'cur_{current_model_correct}_reference_{reference_correct}'

        results = {
            "model_name_or_path": config.model.model_name_or_path,
            "encoder_name_or_path": config.retriever.encoder_name_or_path,
            'avg_recall': np.mean(all_recalls),
            'avg_em_score': np.mean(all_em_scores),
            'avg_cover_em_score': np.mean(all_cover_em_scores),
            'avg_f1_score': np.mean(all_f1_scores),
            'avg_search_success_score': np.mean(all_search_success_scores),
            'avg_first_recall': np.mean(all_first_recall),
            'avg_first_subquery_cosine': np.mean(all_first_subquery_cosine),
            'avg_first_top1_chunk_cosine': np.mean(all_first_top1_chunk_cosine),
            'avg_first_topk_chunk_cosine': np.mean(all_first_topk_chunk_cosine),
            'avg_first_max_pos_chunk_cosine': np.mean(all_first_max_pos_chunk_cosine),
            'confusion_matrix': confusion_matrix,
            'num_samples': len(results),
            'num_prompt_tokens': num_prompt_tokens,
            'num_completion_tokens': num_completion_tokens,
            'avg_time_per_query': avg_time_per_query,
            'avg_infer_time_per_query': avg_infer_time_per_query,
            'avg_infer_time_subquery_steps': avg_infer_time_subquery_steps,
            'avg_infer_time_answer_steps': avg_infer_time_answer_steps,
            'batch_size': batch_size,
            'results': results
        }
        logger.info(f"Input tokens: {num_prompt_tokens}")
        logger.info(f"Output tokens: {num_completion_tokens}")
        logger.info(f"Number of samples: {len(results['results'])}")
        logger.info(f"Average Recall: {results['avg_recall']:.4f}")
        logger.info(f"Average EM score: {results['avg_em_score']:.4f}")
        logger.info(f"Average Cover EM score: {results['avg_cover_em_score']:.4f}")
        logger.info(f"Average F1 score: {results['avg_f1_score']:.4f}")
        logger.info(f"Average Search Success score: {results['avg_search_success_score']:.4f}")
        logger.info(f"Average First Recall: {results['avg_first_recall']:.4f}")
        logger.info(f"Average First Subquery Cosine: {results['avg_first_subquery_cosine']:.4f}")
        logger.info(f"Average First Top-1 Chunk Cosine: {results['avg_first_top1_chunk_cosine']:.4f}")
        logger.info(f"Average First Top-k Chunk Cosine: {results['avg_first_topk_chunk_cosine']:.4f}")
        logger.info(f"Average First Max-pos Chunk Cosine: {results['avg_first_max_pos_chunk_cosine']:.4f}")
        logger.info(f"Confusion Matrix (rows: current model correct/incorrect, columns: reference model correct/incorrect):\n {results['confusion_matrix']}")

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
            
            if getattr(config.data, "save_reference_subquery_embeddings", False) and all_reference_subquery_embeddings is not None:
                reference_subquery_embeddings_save_filename = os.path.splitext(os.path.basename(save_path))[0] + '_reference_subquery_embeddings.npy'
                reference_subquery_embeddings_save_path = os.path.join(os.path.dirname(save_path), reference_subquery_embeddings_save_filename)
                np.save(reference_subquery_embeddings_save_path, np.array(all_reference_subquery_embeddings, dtype=object))
                logger.info(f"Reference subquery embeddings saved to {reference_subquery_embeddings_save_path}")
            
            if getattr(config.data, "save_subquery_embeddings", False) and all_subquery_embeddings is not None:
                subquery_embeddings_save_filename = os.path.splitext(os.path.basename(save_path))[0] + '_subquery_embeddings.npy'
                subquery_embeddings_save_path = os.path.join(os.path.dirname(save_path), subquery_embeddings_save_filename)
                np.save(subquery_embeddings_save_path, np.array(all_subquery_embeddings, dtype=object))
                logger.info(f"Subquery embeddings saved to {subquery_embeddings_save_path}")
            
        # save results to wandb
        wandb_url = config.get('wandb_url', None)
        if wandb_url is not None and use_wandb:
            try:
                wandb_run_path = wandb_url.split('wandb.ai/')[1].split('/runs/')
                wandb_run_path = wandb_run_path[0] + '/' + wandb_run_path[1]
                run = api.run(wandb_run_path)
                eval_results = run.config["evaluate_based_on_reasoning_steps"] if 'evaluate_based_on_reasoning_steps' in run.config else {}
                name = os.path.splitext(os.path.basename(save_path))[0]
                eval_results[name] = {
                    'avg_recall': results['avg_recall'],
                    'avg_em_score': results['avg_em_score'],
                    'avg_cover_em_score': results['avg_cover_em_score'],
                    'avg_f1_score': results['avg_f1_score'],
                    'avg_search_success_score': results['avg_search_success_score'],
                    'avg_first_recall': results['avg_first_recall'],
                    'avg_first_subquery_cosine': results['avg_first_subquery_cosine'],
                    'avg_first_top1_chunk_cosine': results['avg_first_top1_chunk_cosine'],
                    'avg_first_topk_chunk_cosine': results['avg_first_topk_chunk_cosine'],
                    'avg_first_max_pos_chunk_cosine': results['avg_first_max_pos_chunk_cosine'],
                    'confusion_matrix': results['confusion_matrix'],
                    'num_samples': results['num_samples'],
                    'num_prompt_tokens': results['num_prompt_tokens'],
                    'num_completion_tokens': results['num_completion_tokens'],
                    'avg_time_per_query': results['avg_time_per_query'],
                    'batch_size': results['batch_size'],
                }
                run.config["evaluate_based_on_reasoning_steps"] = eval_results
                run.update()
                logger.info(f"Results saved to wandb url: {wandb_url}")
            except Exception as e:
                logger.error(f"Failed to save results to wandb: {e}")

if __name__ == "__main__":
    main()
