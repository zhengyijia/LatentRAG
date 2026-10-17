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

from model.direct_infer import Direct_Infer
from model.few_shot_rag import Few_Shot_RAG
from model.few_shot_rag_cot import Few_Shot_RAG_CoT
from model.ircot import IRCoT
from model.searchr1 import SearchR1
from model.async_vllm_searchr1 import AsyncVLLMSearchR1
from model.autorefine import AutoRefine
from model.hiprag import HipRAG
from model.searcho1 import SearchO1
from model.iterretgen import IterRetGen
from model.r1searcher import R1Searcher
from model.zerosearch import ZeroSearch
from model.deeprag import DeepRAG
from model.latentrag import LatentRAG
from model.sft_model import SFTModel
from model.agentcpm import AgentCPM
from model.infer_inputs_outputs import IterationRecord
from retriever.encoder import Encoder
# from retriever.faiss_retriever_gpu import FaissRetriever, EmptyRetriever
from retriever.torch_retriever_gpu import TorchRetriever, EmptyRetriever
from retriever.rest_retriever import RESTRetriever
from utils import calculate_em_score, calculate_cover_em_score, calculate_f1_score, calculate_search_success_score, flatten_list, unflatten_list

logger = logging.getLogger(__name__)

MODEL_CLASS_DICT = {
    'Direct-Infer': Direct_Infer,
    'Search-R1': SearchR1, 
    'AsyncVLLMSearchR1': AsyncVLLMSearchR1,
    'AutoRefine': AutoRefine,
    'HipRAG': HipRAG,
    'Search-o1': SearchO1, 
    'IterRetGen': IterRetGen,
    'R1Searcher': R1Searcher, 
    'ZeroSearch': ZeroSearch,
    'DeepRAG': DeepRAG,
    'IRCoT': IRCoT,
    'Few-Shot-LLM': Few_Shot_RAG,
    'Few-Shot-RAG': Few_Shot_RAG, 
    'Few-Shot-CoT': Few_Shot_RAG_CoT, 
    'Few-Shot-RAG-CoT': Few_Shot_RAG_CoT, 
    'LatentRAG': LatentRAG,
    'SFTModel': SFTModel,
    'AgentCPM': AgentCPM,
}

def batch_rollout(batch_data, retriever, model, config):
    top_k = config.retriever.top_k

    finished = OrderedDict((query_id, None) for query_id in batch_data['id'])
    if config.model.model_class in ("AgentCPM", "SFTModel", "Direct-Infer", "Search-R1", "AsyncVLLMSearchR1", "AutoRefine", "HipRAG", "Search-o1", "R1Searcher", "ZeroSearch", "LatentRAG", "DeepRAG"):
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

        retrieval_time = retriever_outputs.end_time - retriever_outputs.start_time
        for result, item in zip(retriever_outputs.results, unfinished):
            item.chunk_items.append(result)
            if 'step_wise_time' not in item.extra_info:
                item.extra_info['step_wise_time'] = dict()
            if 'retrieval_time' not in item.extra_info['step_wise_time']:
                item.extra_info['step_wise_time']['retrieval_time'] = []
            item.extra_info['step_wise_time']['retrieval_time'].append(retrieval_time)
            
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
        batch_size = config.data.batch_size
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
            # TODO: for debug
            # reference_encoder = Encoder.load(config, trainable=False, use_lora=False, custom_encoder=True)
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
        logger.info(f"Average LLM inference time per query for {data_path}: {avg_infer_time_per_query:.4f} seconds")
        logger.info(f"Average LLM inference time for subquery steps for {data_path}: {avg_infer_time_subquery_steps:.4f} seconds")
        logger.info(f"Average LLM inference time for answer step for {data_path}: {avg_infer_time_answer_steps:.4f} seconds")

        avg_retrieval_time_per_query = np.mean([
            np.sum(item.extra_info['step_wise_time']['retrieval_time'])
            if (
                'step_wise_time' in item.extra_info
                and 'retrieval_time' in item.extra_info['step_wise_time']
            )
            else 0
            for item in results
        ])
        logger.info(f"Average retrieval time per query for {data_path}: {avg_retrieval_time_per_query:.4f} seconds")

        avg_total_time_per_query = avg_infer_time_per_query + avg_retrieval_time_per_query
        logger.info(f"Average total time per query for {data_path}: {avg_total_time_per_query:.4f} seconds")

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
            'avg_time_per_query': avg_time_per_query,
            'avg_infer_time_per_query': avg_infer_time_per_query,
            'avg_infer_time_subquery_steps': avg_infer_time_subquery_steps,
            'avg_infer_time_answer_steps': avg_infer_time_answer_steps,
            'avg_retrieval_time_per_query': avg_retrieval_time_per_query,
            'avg_total_time_per_query': avg_total_time_per_query,
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

        if hasattr(model, 'base_model') and model.base_model is not None and \
            hasattr(model.base_model, 'get_metrics') and model.base_model.get_metrics() is not None:

            metrics = model.base_model.get_metrics()

            num_accepted_tokens = None
            num_draft_tokens = None
            for metric in metrics:
                if metric.name == "vllm:spec_decode_num_accepted_tokens":
                    num_accepted_tokens = metric.value
                if metric.name == "vllm:spec_decode_num_draft_tokens":
                    num_draft_tokens = metric.value

            if num_accepted_tokens is not None and num_draft_tokens is not None:
                logger.info(f"Speculative decoding metrics: num_accepted_tokens={num_accepted_tokens}, num_draft_tokens={num_draft_tokens}")
                results['num_accepted_tokens'] = num_accepted_tokens
                results['num_draft_tokens'] = num_draft_tokens
                if num_draft_tokens > 0:
                    spec_decode_acceptance_rate = num_accepted_tokens / num_draft_tokens
                    logger.info(f"Speculative decoding ratio: {spec_decode_acceptance_rate:.4f}")
                    results['spec_decode_acceptance_rate'] = spec_decode_acceptance_rate
                else:
                    logger.warning("No draft tokens generated during speculative decoding.")

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
                eval_results = run.config["evaluate"] if 'evaluate' in run.config else {}
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
                    'avg_retrieval_time_per_query': results['avg_retrieval_time_per_query'],
                    'avg_total_time_per_query': results['avg_total_time_per_query'],
                    'batch_size': results['batch_size'],
                }
                run.config["evaluate"] = eval_results
                run.update()
                logger.info(f"Results saved to wandb url: {wandb_url}")
            except Exception as e:
                logger.error(f"Failed to save results to wandb: {e}")

if __name__ == "__main__":
    main()
