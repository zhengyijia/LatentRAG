import os
import logging
import argparse
import random
import jsonlines
import gc
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset
from copy import deepcopy
from dataclasses import asdict
import time
import torch
from string import Template

from llm_tools.safe_loader import AutoConfigSafeLoader
from llm_tools.vllm_offline import VLLMOffline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

guided_json_schema = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
        }, 
        "answer": {
            "type": "string",
        }
    },
    "required": ["question", "answer"]
}

def main():
    parser = argparse.ArgumentParser(description="Generate random questions from the corpus.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the wiki18 corpus file")
    parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
    parser.add_argument("--model_name", type=str, help="Model name for LLM")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=16, help="Max number of sequences for vllm generation")
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for generation")
    parser.add_argument("--num_samples", type=int, default=2000000, help="Sampling the corpus")
    parser.add_argument("--qa_model_gpu_memory_utilization", type=float, default=0.85, help="GPU memory utilization for the QA model when using vllm")
    parser.add_argument("--use_chat_template", action='store_true', help="Whether to use the chat template for question generation")
    parser.add_argument("--max_tokens_per_question", type=int, default=256, help="Max tokens to generate for each question")

    args = parser.parse_args()
    
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        args.save_path = os.path.dirname(args.save_path) + "/DEBUG_" + os.path.basename(args.save_path)
    
    # Ensure max_length does not exceed model's max_position_embeddings if available
    model_conf = AutoConfigSafeLoader.from_pretrained(args.model_path)
    max_position_embeddings = getattr(model_conf, 'max_position_embeddings', None)
    if max_position_embeddings is not None:
        if args.max_length is None or args.max_length > max_position_embeddings:
            args.max_length = max_position_embeddings
            logger.info(f"Setting max_length to model's max_position_embeddings: {args.max_length}")
    
    logger.info("Question generation with the following parameters:")
    logger.info(str(args))
    
    with jsonlines.open(args.file_path) as reader:
        corpus = [obj for obj in tqdm(reader, desc="Loading corpus")]
    
    if args.num_samples != -1:
        random.seed(42)
        sampled_corpus = random.sample(corpus, k=min(args.num_samples, len(corpus)))
    else:
        sampled_corpus = corpus
    
    del corpus
    gc.collect()
    
    model = VLLMOffline(args.model_path, seed=42, 
                        gpu_memory_utilization=args.qa_model_gpu_memory_utilization,
                        max_model_len=args.max_length,
                        max_num_seqs=args.max_num_seqs,
                        dtype=torch.bfloat16)
    tokenizer = model.get_tokenizer()
    
    prompt_template = Template(
        "Generate a question answer pair based on the given passage. The answer should be short (e.g., a single entity, a short phrase, or \"yes\"/\"no\"). Respond with a JSON object that contains a question and an answer, following the JSON schema below:\n\n"
        "```"
        f"{json.dumps(guided_json_schema, separators=(',', ':'))}"
        "```\n\n"
        "Passage:\n\n"
        "$passage"
    )
    
    qa_pairs = []
    for batch_start in tqdm(range(0, len(sampled_corpus), args.batch_size), desc="Generating qa_pairs"):
        batch = sampled_corpus[batch_start:batch_start + args.batch_size]
        batch_passage_ids = [obj['id'] for obj in batch]
        batch_passages = [obj['contents'] for obj in batch]
        
        # construct the prompt for question generation
        all_prompts = [prompt_template.substitute(passage=passage) for passage in batch_passages]
        if args.use_chat_template:
            all_prompts = tokenizer.apply_chat_template(
                    [[{"role": "user", "content": p}] for p in all_prompts], 
                    add_generation_prompt=True, tokenize=False)
        
        all_responses, metadata = model.batch_infer_with_prompt_texts(all_prompts, max_tokens=args.max_tokens_per_question, 
                                                                      json_schema=guided_json_schema, 
                                                                      num_json_retry=1, 
                                                                      skip_special_tokens=True)
        for i, response in enumerate(all_responses):
            try:
                qa_pair = json.loads(response)
            # except:
            except Exception as e:
                triples = []
                logger.warning(f"Could not parse response from QA pair generation: {e}. Response: {response}")
                continue
            
            passage_id = batch_passage_ids[i]
            qa_pair['passage_id'] = passage_id
            qa_pairs.append(qa_pair)
    
    qa_pairs = [
        {
            "id": f"pretrain_{i}", 
            "question": qa_pair["question"].strip(),
            "golden_answers": [qa_pair["answer"].strip()],
            "passage_id": qa_pair["passage_id"]
        }
        for i, qa_pair in enumerate(qa_pairs)
    ]
    
    save_dirname = os.path.dirname(args.save_path)
    if not os.path.exists(save_dirname):
        os.makedirs(save_dirname)
    
    with jsonlines.open(args.save_path, mode='w') as writer:
        writer.write_all(qa_pairs)
        
    logger.info(f"Saved {len(qa_pairs)} QA pairs to {args.save_path}")

if __name__ == "__main__":
    main()
