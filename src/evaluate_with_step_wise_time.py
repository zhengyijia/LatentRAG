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
from model.autorefine import AutoRefine
from model.few_shot_rag import Few_Shot_RAG
from model.sft_model import SFTModel
from model.agentcpm import AgentCPM
# from model.agentcpm_with_repetition_penalty import AgentCPM
from model.latentrag import LatentRAG
from model.infer_inputs_outputs import IterationRecord
from retriever.encoder import Encoder
# from retriever.faiss_retriever_gpu import FaissRetriever, EmptyRetriever
from retriever.torch_retriever_gpu import TorchRetriever, EmptyRetriever
from retriever.rest_retriever import RESTRetriever
from utils import calculate_em_score, calculate_cover_em_score, calculate_f1_score, calculate_search_success_score, flatten_list, unflatten_list

logger = logging.getLogger(__name__)

MODEL_CLASS_DICT = {
    'Search-R1': SearchR1, 
    'AutoRefine': AutoRefine,
    'Few-Shot-RAG': Few_Shot_RAG,
    'SFTModel': SFTModel,
    'AgentCPM': AgentCPM,
    'LatentRAG': LatentRAG,
}

def batch_rollout(batch_data, retriever, model, config):
    top_k = config.retriever.top_k

    finished = OrderedDict((query_id, None) for query_id in batch_data['id'])
    if config.model.model_class in ("Search-R1", "AutoRefine", "SFTModel", "AgentCPM", "LatentRAG"):
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
        retriever_outputs = retriever.retrieve_with_text(questions, top_k=top_k)
        inputs = [
            IterationRecord(
                query_id = query_id,
                query = question,
                golden_answers = [gold_answers] if isinstance(gold_answers, str) else gold_answers,
                chunk_items = [chunk_items],
            )
            for query_id, question, gold_answers, chunk_items in zip(
                batch_data['id'], batch_data['question'], batch_data['golden_answers'], retriever_outputs.results
            )
        ]
        
        retrieval_time = retriever_outputs.end_time - retriever_outputs.start_time
        for item in inputs:
            item.extra_info['step_wise_time'] = dict()
            item.extra_info['step_wise_time']['retrieval_time'] = []
            item.extra_info['step_wise_time']['retrieval_time'].append(retrieval_time)

    num_prompt_tokens = 0
    num_completion_tokens = 0
    while True: 
        kwargs = {}
        if config.get('batch_subquery_decoding', False):
            kwargs['batch_subquery_decoding'] = True
        
        outputs, metadata = model.generate_with_step_wise_time(inputs, **kwargs)
        
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
        
        retrieval_time = retriever_outputs.end_time - retriever_outputs.start_time
        for item in unfinished:
            if 'step_wise_time' not in item.extra_info:
                item.extra_info['step_wise_time'] = dict()
            if 'retrieval_time' not in item.extra_info['step_wise_time']:
                item.extra_info['step_wise_time']['retrieval_time'] = []
            item.extra_info['step_wise_time']['retrieval_time'].append(retrieval_time)
            
        for result, item in zip(retriever_outputs.results, unfinished):
            item.chunk_items.append(result)
            
        inputs = unfinished

    return finished.values(), {'prompt_tokens': num_prompt_tokens, 'completion_tokens': num_completion_tokens}


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
        # batch_size = config.data.batch_size
        batch_size = 1 # fix batch_size to 1 for recording request latency per sample
        file_prefix += f"_bs{batch_size}"
        if config.vllm.get('use_vllm', False):
            file_prefix += f"_use_vllm"
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
        dataset = load_dataset("json", data_files={split: data_path})[split]
        if config.data.get('sampling', -1) > 0:
            sampling = min(config.data.sampling, len(dataset))
            dataset = dataset.shuffle(seed=42).select(range(sampling))
            logger.info(f"Using {sampling} samples from {data_path}")

        results = []
        num_prompt_tokens = 0
        num_completion_tokens = 0
        batch_size = 1 # fix batch_size to 1 for recording request latency per sample
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

        # avg_infer_time_per_query = np.mean([np.sum(item.infer_time_each_steps) for item in results])
        # avg_infer_time_subquery_steps = np.mean([np.sum(item.infer_time_each_steps[:-1]) for item in results if len(item.infer_time_each_steps) > 1])
        # avg_infer_time_answer_steps = np.mean([item.infer_time_each_steps[-1] for item in results])
        # logger.info(f"Average inference time per query for {data_path}: {avg_infer_time_per_query:.4f} seconds")
        # logger.info(f"Average inference time for subquery steps for {data_path}: {avg_infer_time_subquery_steps:.4f} seconds")
        # logger.info(f"Average inference time for answer step for {data_path}: {avg_infer_time_answer_steps:.4f} seconds")

        # embeddings cannot be serialized
        all_query_embeddings = [item.query_embedding for item in results]
        all_subquery_embeddings = [item.subquery_embeddings for item in results]
        for item in results:
            item.query_embedding = None
            item.subquery_embeddings = None
        results = [asdict(item) for item in results]
        
        all_em_scores = []
        all_cover_em_scores = []
        all_f1_scores = []
        all_search_success_scores = []
        all_prefill_time = []
        all_thoughts_generation_time = []
        all_subqueries_generation_time = []
        all_answer_generation_time = []
        all_retrieval_time = []
        all_batch_decoding_time = []
        for i, item in enumerate(results):
            golden_answers = item['golden_answers']
            predicted_answer = item['answer']
            em_score = calculate_em_score(golden_answers, predicted_answer)
            cover_em_score = calculate_cover_em_score(golden_answers, predicted_answer)
            f1_score = calculate_f1_score(golden_answers, predicted_answer)
            search_success = calculate_search_success_score(golden_answers, [chunk_item['text'] for chunk_item in flatten_list(item['chunk_items'])])
            item['em_score'] = em_score
            item['cover_em_score'] = cover_em_score
            item['f1_score'] = f1_score
            item['search_success'] = search_success
            all_em_scores.append(em_score)
            all_cover_em_scores.append(cover_em_score)
            all_f1_scores.append(f1_score)
            all_search_success_scores.append(search_success)
            
            step_wise_time = item['extra_info'].get('step_wise_time', {})
            prefill_time = thoughts_generation_time = subqueries_generation_time = answer_generation_time = retrieval_time = batch_decoding_time = 0.0
            if 'prefill_time' in step_wise_time:
                prefill_time = sum([record_time for record_time in step_wise_time['prefill_time'] if record_time is not None])
            if 'thought_generation_time' in step_wise_time:
                thoughts_generation_time = sum([record_time for record_time in step_wise_time['thought_generation_time'] if record_time is not None])
            if 'subquery_generation_time' in step_wise_time:
                subqueries_generation_time = sum([record_time for record_time in step_wise_time['subquery_generation_time'] if record_time is not None])
            if 'answer_generation_time' in step_wise_time:
                answer_generation_time = sum([record_time for record_time in step_wise_time['answer_generation_time'] if record_time is not None])
            if 'retrieval_time' in step_wise_time:
                retrieval_time = sum([record_time for record_time in step_wise_time['retrieval_time'] if record_time is not None])
            if 'batch_decoding_time' in step_wise_time:
                batch_decoding_time = sum([record_time for record_time in step_wise_time['batch_decoding_time'] if record_time is not None])
            all_prefill_time.append(prefill_time)
            all_thoughts_generation_time.append(thoughts_generation_time)
            all_subqueries_generation_time.append(subqueries_generation_time)
            all_answer_generation_time.append(answer_generation_time)
            all_retrieval_time.append(retrieval_time)
            all_batch_decoding_time.append(batch_decoding_time)
            
        results = {
            "model_name_or_path": config.model.model_name_or_path,
            "encoder_name_or_path": config.retriever.encoder_name_or_path,
            'avg_em_score': np.mean(all_em_scores),
            'avg_cover_em_score': np.mean(all_cover_em_scores),
            'avg_f1_score': np.mean(all_f1_scores),
            'avg_search_success_score': np.mean(all_search_success_scores),
            'avg_prefill_time': np.mean(all_prefill_time),
            'avg_thoughts_generation_time': np.mean(all_thoughts_generation_time),
            'avg_subqueries_generation_time': np.mean(all_subqueries_generation_time),
            'avg_answer_generation_time': np.mean(all_answer_generation_time),
            'avg_retrieval_time': np.mean(all_retrieval_time),
            'avg_batch_decoding_time': np.mean(all_batch_decoding_time),
            'num_samples': len(results),
            'num_prompt_tokens': num_prompt_tokens,
            'num_completion_tokens': num_completion_tokens,
            'avg_time_per_query': avg_time_per_query,
            'batch_size': batch_size,
            'results': results
        }
        logger.info(f"Input tokens: {num_prompt_tokens}")
        logger.info(f"Output tokens: {num_completion_tokens}")
        logger.info(f"Number of samples: {len(results['results'])}")
        logger.info(f"Average EM score: {results['avg_em_score']:.4f}")
        logger.info(f"Average Cover EM score: {results['avg_cover_em_score']:.4f}")
        logger.info(f"Average F1 score: {results['avg_f1_score']:.4f}")
        logger.info(f"Average Search Success score: {results['avg_search_success_score']:.4f}")
        logger.info(f"Average Prefill Time: {results['avg_prefill_time']:.4f}")
        logger.info(f"Average Thoughts Generation Time: {results['avg_thoughts_generation_time']:.4f}")
        logger.info(f"Average Subqueries Generation Time: {results['avg_subqueries_generation_time']:.4f}")
        logger.info(f"Average Answer Generation Time: {results['avg_answer_generation_time']:.4f}")
        logger.info(f"Average Retrieval Time: {results['avg_retrieval_time']:.4f}")
        logger.info(f"Average Batch Decoding Time: {results['avg_batch_decoding_time']:.4f}")
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
            
        # save results to wandb
        wandb_url = config.get('wandb_url', None)
        if wandb_url is not None and use_wandb:
            try:
                wandb_run_path = wandb_url.split('wandb.ai/')[1].split('/runs/')
                wandb_run_path = wandb_run_path[0] + '/' + wandb_run_path[1]
                run = api.run(wandb_run_path)
                eval_results = run.config["evaluate_with_step_wise_time"] if 'evaluate_with_step_wise_time' in run.config else {}
                name = os.path.splitext(os.path.basename(save_path))[0]
                eval_results[name] = {
                    'model_name_or_path': config.model.model_name_or_path,
                    "encoder_name_or_path": config.retriever.encoder_name_or_path,
                    'avg_em_score': results['avg_em_score'],
                    'avg_cover_em_score': results['avg_cover_em_score'],
                    'avg_f1_score': results['avg_f1_score'],
                    'avg_search_success_score': results['avg_search_success_score'],
                    'num_samples': results['num_samples'],
                    'num_prompt_tokens': results['num_prompt_tokens'],
                    'num_completion_tokens': results['num_completion_tokens'],
                    'avg_time_per_query': results['avg_time_per_query'],
                    'batch_size': results['batch_size'],
                    'avg_prefill_time': results['avg_prefill_time'],
                    'avg_thoughts_generation_time': results['avg_thoughts_generation_time'],
                    'avg_subqueries_generation_time': results['avg_subqueries_generation_time'],
                    'avg_answer_generation_time': results['avg_answer_generation_time'],
                    'avg_retrieval_time': results['avg_retrieval_time'],
                    'avg_batch_decoding_time': results['avg_batch_decoding_time'],
                }
                run.config["evaluate_with_step_wise_time"] = eval_results
                run.update()
                logger.info(f"Results saved to wandb url: {wandb_url}")
            except Exception as e:
                logger.error(f"Failed to save results to wandb: {e}")

if __name__ == "__main__":
    main()
