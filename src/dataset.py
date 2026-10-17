import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from tqdm import tqdm
import random
from itertools import accumulate
from copy import deepcopy
import jsonlines
import gc

from model.latentrag import LatentRAG
from utils import log_example_data

import logging
logger = logging.getLogger(__name__)

def get_dataset(files, config, tokenizer, 
                batch_size, 
                eval_batch_size=None, # only used when split_train_eval_set is True
                shuffle=True, filter_em_correct=True, filter_cover_em_correct=False, filter_llm_judge_correct=False,
                split_train_eval_set=False, num_eval_samples=200, 
                num_train_samples=-1):

    def _process_chunk_texts(texts, title_in_first_line=True): 
        format_reference = ''
        for idx, text in enumerate(texts):
            if title_in_first_line:
                title = text.split("\n")[0]
                text = "\n".join(text.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            else:
                format_reference += f"[Doc {idx+1}]\n{text}\n"

        return format_reference

    def _load_dataset(files, config):
        title_in_first_line = getattr(config.data, 'title_in_first_line', True)
        
        data = []
        for file in files:
            file_path = os.path.join(config.data.data_path, file)
            reasoning_steps_data = json.load(open(file_path))['results']
            logger.info(f"Loaded {len(reasoning_steps_data)} reasoning paths data from {file_path}.")
            
            for item in reasoning_steps_data:
                item['source_file'] = file
                
            data.extend(reasoning_steps_data)

        if filter_em_correct:
            data = [item for item in data if item['em_score']==1.0]
            
        if filter_cover_em_correct:
            data = [item for item in data if item['cover_em_score']==1.0]

        if filter_llm_judge_correct:
            data = [item for item in data if item['llm_judge_result']['correct']==True]

        all_chunk_contains_text = all(all((('text' in chunk_item) and chunk_item['text']) for chunk_item in chunk_items) for item in data for chunk_items in item['chunk_items'])
        all_hard_neg_chunk_contains_text = all(all((('text' in chunk_item) and chunk_item['text']) for chunk_item in chunk_items) for item in data for chunk_items in item['hard_neg_chunk_items'])
        if not all_chunk_contains_text or not all_hard_neg_chunk_contains_text:
            logger.info("Some chunk items do not contain 'text' field. Load text from corpus...")
            corpus_file_path = os.path.join(config.retriever.corpus_path,
                                            config.retriever.corpus_file)
            with jsonlines.open(corpus_file_path) as reader:
                corpus_data = {obj['id']: obj for obj in tqdm(reader, desc="Loading corpus")}
            
            for item in tqdm(data, desc="Loading chunk texts"):
                for chunk_items in item['chunk_items']:
                    for chunk_item in chunk_items:
                        if 'text' not in chunk_item or not chunk_item['text']:
                            chunk_id = chunk_item['chunk_id']
                            chunk_item['text'] = corpus_data[chunk_id]['contents']
                for chunk_items in item['hard_neg_chunk_items']:
                    for chunk_item in chunk_items:
                        if 'text' not in chunk_item or not chunk_item['text']:
                            chunk_id = chunk_item['chunk_id']
                            chunk_item['text'] = corpus_data[chunk_id]['contents']
            
            del corpus_data  # free memory
            gc.collect()
            
            logger.info("Finished loading chunk texts.")
            
        
        if config.model.model_class == "LatentRAG":
            model_cls = LatentRAG
        else:
            raise ValueError(f"Unknown model_class: {config.model.model_class}")

        spec_tokens, _ = model_cls.get_special_added_tokens(config)
        spec_token_ids = {name: tokenizer.convert_tokens_to_ids(token)
                          for name, token in spec_tokens.items()}
        spec_prompts = model_cls.get_special_prompts(config)
        
        # all_prompts = [spec_tokens['boq_token'] + item["query"] + spec_tokens['eoq_token'] for item in data]
        # self_decode_thought_prompt = spec_tokens['bode_thought_token']  + ''.join(spec_tokens['latent_tokens']) + spec_tokens['eode_thought_token']
        # self_decode_subquery_prompt = spec_tokens['bode_subquery_token'] + ''.join(spec_tokens['latent_subquery_tokens']) + spec_tokens['eode_subquery_token']
        all_prompts = [spec_prompts['query_prompt'].format(query=item["query"]) for item in data]
        self_decode_thought_prompt = spec_prompts['self_decode_thought_prompt']
        self_decode_subquery_prompt = spec_prompts['self_decode_subquery_prompt']
        if config.model.use_chat_template:
            all_prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in all_prompts], 
                                                        add_generation_prompt=True, tokenize=False)
            self_decode_thought_prompt = tokenizer.apply_chat_template([[{"role": "user", "content": self_decode_thought_prompt}]], 
                                                                       add_generation_prompt=True, tokenize=False)[0]
            self_decode_subquery_prompt = tokenizer.apply_chat_template([[{"role": "user", "content": self_decode_subquery_prompt}]], 
                                                                        add_generation_prompt=True, tokenize=False)[0]
        
        for i, item in enumerate(tqdm(data, desc="Processing data")):
            # only preserve thoughts for generating subqueries
            item['thoughts'] = [thought.strip() if thought is not None else "" for thought in item['thoughts']]
            item['subqueries'] = [subquery.strip() for subquery in item['subqueries'] if subquery is not None]
            item['answer'] = item['answer'].strip()
            assert len(item['thoughts'])-1 == len(item['subqueries']) == len(item['chunk_items'])
            
            # construct input text
            input_text = all_prompts[i]
            
            for j, subquery in enumerate(item['subqueries']):
                input_text += ''.join(spec_tokens['latent_tokens'])
                input_text += ''.join(spec_tokens['latent_subquery_tokens'])
                
                input_text += spec_tokens['bod_token']
                input_text += _process_chunk_texts([chunk_item['text'] for chunk_item in item['chunk_items'][j]], title_in_first_line=title_in_first_line)
                input_text += spec_tokens['eod_token']
                
            input_text += ''.join(spec_tokens['latent_tokens'])
            
            input_text += spec_tokens['boa_token']
            input_text += item['answer']
            input_text += spec_tokens['eoa_token']
            input_text += tokenizer.eos_token
            
            item['input_text'] = input_text
            
            # construct self decode input texts
            decode_thought = getattr(config.custom_training, 'decode_thought', False)
            if decode_thought:
                item['self_decode_thought_input_texts'] = [
                    self_decode_thought_prompt + 
                    spec_tokens['bot_token'] + thought + spec_tokens['eot_token'] + 
                    tokenizer.eos_token
                    for thought in item['thoughts']
                ]
            else:
                item['self_decode_thought_input_texts'] = []
                
            decode_subquery = getattr(config.custom_training, 'decode_subquery', False)
            if decode_subquery:
                item['self_decode_subquery_input_texts'] = [
                    self_decode_subquery_prompt + 
                    spec_tokens['bosq_token'] + subquery + spec_tokens['eosq_token'] + 
                    tokenizer.eos_token
                    for subquery in item['subqueries']
                ]
            else:
                item['self_decode_subquery_input_texts'] = []
        
        # construct input_ids
        all_input_texts = [item['input_text'] for item in data]
        all_input_token_ids = tokenizer(
            all_input_texts,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )['input_ids']
        max_token_length = max(len(input_token_ids) for input_token_ids in all_input_token_ids)
        logger.info(f"Max token length in dataset: {max_token_length}")
        # logging sequence length distribution
        larger_than_4000 = len([len(input_token_ids) for input_token_ids in all_input_token_ids if len(input_token_ids) > 4000])
        logger.info(f"Number of samples with input_ids length larger than 4000: {larger_than_4000}")
        larger_than_3500 = len([len(input_token_ids) for input_token_ids in all_input_token_ids if len(input_token_ids) > 3500])
        logger.info(f"Number of samples with input_ids length larger than 3500: {larger_than_3500}")
        larger_than_3000 = len([len(input_token_ids) for input_token_ids in all_input_token_ids if len(input_token_ids) > 3000])
        logger.info(f"Number of samples with input_ids length larger than 3000: {larger_than_3000}")
        larger_than_2500 = len([len(input_token_ids) for input_token_ids in all_input_token_ids if len(input_token_ids) > 2500])
        logger.info(f"Number of samples with input_ids length larger than 2500: {larger_than_2500}")
        larger_than_2000 = len([len(input_token_ids) for input_token_ids in all_input_token_ids if len(input_token_ids) > 2000])
        logger.info(f"Number of samples with input_ids length larger than 2000: {larger_than_2000}")
        longer_than_1500 = len([len(input_token_ids) for input_token_ids in all_input_token_ids if len(input_token_ids) > 1500])
        logger.info(f"Number of samples with input_ids length larger than 1500: {longer_than_1500}")
        longer_than_1000 = len([len(input_token_ids) for input_token_ids in all_input_token_ids if len(input_token_ids) > 1000])
        logger.info(f"Number of samples with input_ids length larger than 1000: {longer_than_1000}")
        
        for i, item in enumerate(data):
            item['input_ids'] = all_input_token_ids[i]

        max_seq_length = config.data.get('max_seq_length', 3000)
        data = [item for item in data if len(item['input_ids']) <= max_seq_length]
        logger.info(f"Filtered dataset size after max_seq_length {max_seq_length}: {len(data)}")
            
        # construct labels
        for item in data:
            boa_pos = item['input_ids'].index(spec_token_ids['boa_token'])
            item['labels'] = [-100] * (boa_pos+1) + item['input_ids'][boa_pos+1:]
            if not item['force_answer']: # predict the answer without boa_token
                item['labels'][boa_pos] = spec_token_ids['boa_token']
            for i, token_id in enumerate(item['input_ids']):
                if token_id == spec_token_ids['latent_subquery_begin_token']: 
                    item['labels'][i] = spec_token_ids['bosq_token']
                    
        # construct self decode thought input_ids
        flat_all_self_decode_thought_input_texts = [text for item in data for text in item['self_decode_thought_input_texts']]
        if len(flat_all_self_decode_thought_input_texts) > 0:
            flat_all_self_decode_thought_input_token_ids = tokenizer(
                flat_all_self_decode_thought_input_texts,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )['input_ids']
            max_token_length = max(len(input_token_ids) for input_token_ids in flat_all_self_decode_thought_input_token_ids)
            logger.info(f"Max self decode thought token length in dataset: {max_token_length}")
            lengths = [len(item['self_decode_thought_input_texts']) for item in data]
            offsets = [0] + list(accumulate(lengths))
            unflat_all_self_decode_thought_input_token_ids = [
                flat_all_self_decode_thought_input_token_ids[offsets[i]:offsets[i+1]]
                for i in range(len(lengths))
            ]
            for i, item in enumerate(data):
                item['self_decode_thought_input_ids_list'] = unflat_all_self_decode_thought_input_token_ids[i]
        else:
            for item in data:
                item['self_decode_thought_input_ids_list'] = []

        # construct self decode thought labels
        decode_begin_token_id = spec_token_ids['bot_token']
        for item in data:
            self_decode_thought_labels_list = []
            for self_decode_thought_input_ids in item['self_decode_thought_input_ids_list']:
                decode_begin_pos = next(i for i, v in enumerate(self_decode_thought_input_ids) if v == decode_begin_token_id)
                self_decode_thought_labels = [-100] * (decode_begin_pos) + self_decode_thought_input_ids[decode_begin_pos:]
                
                # do not generate empty thoughts, subqueries
                if self_decode_thought_input_ids[decode_begin_pos + 1] == spec_token_ids['eot_token']:
                    self_decode_thought_labels[decode_begin_pos + 1] = -100
                
                self_decode_thought_labels_list.append(self_decode_thought_labels)
            item['self_decode_thought_labels_list'] = self_decode_thought_labels_list
            
        # construct self decode subquery input_ids
        flat_all_self_decode_subquery_input_texts = [text for item in data for text in item['self_decode_subquery_input_texts']]
        if len(flat_all_self_decode_subquery_input_texts) > 0:
            flat_all_self_decode_subquery_input_token_ids = tokenizer(
                flat_all_self_decode_subquery_input_texts,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )['input_ids']
            max_token_length = max(len(input_token_ids) for input_token_ids in flat_all_self_decode_subquery_input_token_ids)
            logger.info(f"Max self decode subquery token length in dataset: {max_token_length}")
            lengths = [len(item['self_decode_subquery_input_texts']) for item in data]
            offsets = [0] + list(accumulate(lengths))
            unflat_all_self_decode_subquery_input_token_ids = [
                flat_all_self_decode_subquery_input_token_ids[offsets[i]:offsets[i+1]]
                for i in range(len(lengths))
            ]
            for i, item in enumerate(data):
                item['self_decode_subquery_input_ids_list'] = unflat_all_self_decode_subquery_input_token_ids[i]
        else:
            for item in data:
                item['self_decode_subquery_input_ids_list'] = []
                
        # construct self decode subquery labels
        decode_begin_token_id = spec_token_ids['bosq_token']
        for item in data:
            self_decode_subquery_labels_list = []
            for self_decode_subquery_input_ids in item['self_decode_subquery_input_ids_list']:
                decode_begin_pos = next(i for i, v in enumerate(self_decode_subquery_input_ids) if v == decode_begin_token_id)
                self_decode_subquery_labels = [-100] * (decode_begin_pos) + self_decode_subquery_input_ids[decode_begin_pos:]
                
                # do not generate empty thoughts, subqueries
                if self_decode_subquery_input_ids[decode_begin_pos + 1] == spec_token_ids['eosq_token']:
                    self_decode_subquery_labels[decode_begin_pos + 1] = -100
                
                self_decode_subquery_labels_list.append(self_decode_subquery_labels)
            item['self_decode_subquery_labels_list'] = self_decode_subquery_labels_list

        return data

    if isinstance(files, str):
        files = [files]  # single file
    if dist.is_initialized():
        if dist.get_rank() == 0:
            data = _load_dataset(files, config)
            data = [data]
        else:
            data = [None]
        dist.barrier()
        dist.broadcast_object_list(data, src=0)
        data = data[0]
    else:
        data = _load_dataset(files, config)

    if split_train_eval_set:
        assert num_eval_samples < len(data), f"num_eval_samples {num_eval_samples} should be smaller than dataset size {len(data)}."
        
        rng = random.Random(config.training.get('seed', 42))
        indices = list(range(len(data)))
        rng.shuffle(indices)
        train_indices = indices[:-num_eval_samples]
        eval_indices = indices[-num_eval_samples:]
        train_data = [data[i] for i in train_indices]
        eval_data = [data[i] for i in eval_indices]
        logger.info(f"Split dataset into train size: {len(train_data)} and eval size: {len(eval_data)}")

        num_train_samples = min(num_train_samples, len(train_data)) if num_train_samples > 0 else len(train_data)
        train_data = train_data[:num_train_samples]
        logger.info(f"Using {num_train_samples} samples for training.")

        return (SFTDataset(data=train_data, tokenizer=tokenizer, config=config, batch_size=batch_size, 
                           shuffle=shuffle),
                SFTDataset(data=eval_data, tokenizer=tokenizer, config=config, batch_size=eval_batch_size, 
                           shuffle=False))

    if num_train_samples > 0:
        num_train_samples = min(num_train_samples, len(data))
        data = data[:num_train_samples]
        logger.info(f"Using {num_train_samples} samples for training.")

    return SFTDataset(data=data, tokenizer=tokenizer, config=config, batch_size=batch_size, 
                      shuffle=shuffle)
    

class SFTDataset(Dataset):
    def __init__(self,
                 data, 
                 tokenizer, 
                 config, 
                 batch_size, 
                 shuffle=True):

        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # for nce loss training
        self.pos_k = config.retriever.get('top_k', 3)
        self.neg_k = config.custom_training.get('neg_k', 0)
        
        self.decode_thought = getattr(config.custom_training, 'decode_thought', False)
        self.decode_subquery = getattr(config.custom_training, 'decode_subquery', False)
        
        if config.model.model_class == "LatentRAG":
            model_cls = LatentRAG
        else:
            raise ValueError(f"Unknown model_class: {config.model.model_class}")

        self.spec_tokens, _ = model_cls.get_special_added_tokens(config)
        self.spec_token_ids = {name: tokenizer.convert_tokens_to_ids(token)
                               for name, token in self.spec_tokens.items()}
        
        self.seed = config.training.get('seed', 42)
        self.n_bins = config.data.get('n_bins', 10)
        self.label_pad_token_id=-100

        if self.shuffle:
            data = sorted(data, key=lambda x: len(x['input_ids']))  # sort by length to minimize padding
        self.data = data
        
        if self.shuffle:
            self.shuffle_and_rebatch(epoch=0)
        else:
            self.all_batch_indices = [list(range(i, min(i + self.batch_size, len(self.data)))) 
                                      for i in range(0, len(self.data), self.batch_size)]
            # all_batch_data = [[self.data[idx] for idx in batch_indices] for batch_indices in all_batch_indices]
            # self.all_batch_data = [self.process_batch_data(batch_data) for batch_data in tqdm(all_batch_data)]

        # for logging example data
        batch_indices = self.all_batch_indices[0]
        batch_data = [self.data[idx] for idx in batch_indices]
        batch_data = self.process_batch_data(batch_data)
        logger.info("######## Logging example data ########")
        log_example_data(batch_data[0]['input_batch'], self.tokenizer, self.label_pad_token_id)
        if self.decode_thought:
            logger.info("######## Logging example thought self decoding data ########")
            log_example_data(batch_data[0]['self_decode_thought_input_batch'], self.tokenizer, self.label_pad_token_id)
        if self.decode_subquery:
            logger.info("######## Logging example subquery self decoding data ########")
            log_example_data(batch_data[0]['self_decode_subquery_input_batch'], self.tokenizer, self.label_pad_token_id)

    def __len__(self):
        return len(self.all_batch_indices)
    
    def shuffle_and_rebatch(self, epoch: int = 0):
        """Call this function at the beginning of each epoch to reshuffle the data and create new batches."""
        seed = self.seed + epoch  # Ensure different randomness for each epoch
        rng = random.Random(seed)

        # shuffle batch data
        bin_size = (len(self.data) + self.n_bins - 1) // self.n_bins
        bins = [list(range(i, min(i + bin_size, len(self.data)))) for i in range(0, len(self.data), bin_size)]
        for b in bins:
            rng.shuffle(b)
        shuffled_indices = [idx for b in bins for idx in b]
        self.all_batch_indices = [shuffled_indices[i:i + self.batch_size] for i in range(0, len(shuffled_indices), self.batch_size)]
        # all_batch_data = [[self.data[idx] for idx in batch_indices] for batch_indices in all_batch_indices]
        # self.all_batch_data = [self.process_batch_data(batch_data) for batch_data in tqdm(all_batch_data)]
        # sum_data_len = sum([batch_data[0]['input_ids'].shape[1] for batch_data in self.all_batch_data])
        # logger.info(f"After shuffling, number of batches: {len(self.all_batch_data)}, sum data length in batches: {sum_data_len}")
    
    def process_batch_data(self, batch_data):
        input_features = [{'input_ids': item['input_ids'], 'labels': item['labels']} for item in batch_data]
        all_query_ids = [item['query_id'] for item in batch_data]
        all_queries = [item['query'] for item in batch_data]
        all_thoughts = [item['thoughts'] for item in batch_data]
        all_subqueries = [item['subqueries'] for item in batch_data]
        self_decode_thought_input_features = [{'input_ids': input_ids, 'labels': labels} 
                                              for item in batch_data 
                                              for input_ids, labels in zip(item['self_decode_thought_input_ids_list'], item['self_decode_thought_labels_list'])]
        self_decode_subquery_input_features = [{'input_ids': input_ids, 'labels': labels} 
                                              for item in batch_data 
                                              for input_ids, labels in zip(item['self_decode_subquery_input_ids_list'], item['self_decode_subquery_labels_list'])]
        
        all_pos_chunk_items = [item['chunk_items'] for item in batch_data]
        # make sure each chunk_items has exact pos_k items
        all_pos_chunk_items = [
            [
                (
                    chunk_items[:self.pos_k] # preserve the first pos_k items (with highest scores)
                    if len(chunk_items) >= self.pos_k 
                    else (chunk_items + random.choices(chunk_items, k=self.pos_k - len(chunk_items)))
                )
                for chunk_items in sublist
            ]
            for sublist in all_pos_chunk_items
        ]

        all_hard_neg_chunk_items = [item['hard_neg_chunk_items'] for item in batch_data]
        # make sure each chunk_items has exact neg_k items
        all_hard_neg_chunk_items = [
            [
                (
                    random.sample(chunk_items, self.neg_k) if len(chunk_items) >= self.neg_k
                    else (chunk_items + random.choices(chunk_items, k=self.neg_k - len(chunk_items)))
                )
                for chunk_items in sublist
            ]
            for sublist in all_hard_neg_chunk_items
        ]

        # construct LLM batch input
        for feature in input_features:
            feature['attention_mask'] = [1] * len(feature['input_ids'])
            feature['position_ids'] = list(range(len(feature['input_ids'])))
            
        # only left padding
        assert self.tokenizer.padding_side == "left"
        
        non_label_position_input_features = [
            {
                k: v
                for k, v in feature.items()
                if k != "labels" and k != "position_ids"
            }
            for feature in input_features
        ]
        # run through tokenizer without labels to ensure no side effects
        input_batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_input_features,
            padding=True,
            return_tensors='pt',
        )
        labels = [feature['labels'] for feature in input_features]
        position_ids = [feature["position_ids"] for feature in input_features]
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`
        input_ids_length = input_batch['input_ids'].shape[-1]
        input_batch["labels"] = [
            [self.label_pad_token_id] * (input_ids_length - len(label)) + label
            for label in labels
        ]
        input_batch["labels"] = torch.tensor(input_batch["labels"], dtype=torch.int64)
        input_batch["position_ids"] = [
            [0] * (input_ids_length - len(position_id)) + position_id
            for position_id in position_ids
        ]
        input_batch["position_ids"] = torch.tensor(input_batch["position_ids"], dtype=torch.int64)

        # construct self decode thought LLM batch input
        self_decode_thought_input_batch = None
        if len(self_decode_thought_input_features) > 0:
            for feature in self_decode_thought_input_features:
                feature['attention_mask'] = [1] * len(feature['input_ids'])
                feature['position_ids'] = list(range(len(feature['input_ids'])))

            # only left padding
            assert self.tokenizer.padding_side == "left"
            
            non_label_position_self_decode_thought_input_features = [
                {
                    k: v
                    for k, v in feature.items()
                    if k != "labels" and k != "position_ids"
                }
                for feature in self_decode_thought_input_features
            ]
            # run through tokenizer without labels to ensure no side effects
            self_decode_thought_input_batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                non_label_position_self_decode_thought_input_features,
                padding=True,
                return_tensors='pt',
            )
            self_decode_thought_labels = [feature['labels'] for feature in self_decode_thought_input_features]
            self_decode_thought_position_ids = [feature["position_ids"] for feature in self_decode_thought_input_features]
            # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`
            self_decode_thought_input_ids_length = self_decode_thought_input_batch['input_ids'].shape[-1]
            self_decode_thought_input_batch["labels"] = [
                [self.label_pad_token_id] * (self_decode_thought_input_ids_length - len(label)) + label
                for label in self_decode_thought_labels
            ]
            self_decode_thought_input_batch["labels"] = torch.tensor(self_decode_thought_input_batch["labels"], dtype=torch.int64)
            self_decode_thought_input_batch["position_ids"] = [
                [0] * (self_decode_thought_input_ids_length - len(position_id)) + position_id
                for position_id in self_decode_thought_position_ids
            ]
            self_decode_thought_input_batch["position_ids"] = torch.tensor(self_decode_thought_input_batch["position_ids"], dtype=torch.int64)

        # construct self decode subquery LLM batch input
        self_decode_subquery_input_batch = None
        if len(self_decode_subquery_input_features) > 0:
            for feature in self_decode_subquery_input_features:
                feature['attention_mask'] = [1] * len(feature['input_ids'])
                feature['position_ids'] = list(range(len(feature['input_ids'])))

            # only left padding
            assert self.tokenizer.padding_side == "left"
            
            non_label_position_self_decode_subquery_input_features = [
                {
                    k: v
                    for k, v in feature.items()
                    if k != "labels" and k != "position_ids"
                }
                for feature in self_decode_subquery_input_features
            ]
            # run through tokenizer without labels to ensure no side effects
            self_decode_subquery_input_batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                non_label_position_self_decode_subquery_input_features,
                padding=True,
                return_tensors='pt',
            )
            self_decode_subquery_labels = [feature['labels'] for feature in self_decode_subquery_input_features]
            self_decode_subquery_position_ids = [feature["position_ids"] for feature in self_decode_subquery_input_features]
            # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`
            self_decode_subquery_input_ids_length = self_decode_subquery_input_batch['input_ids'].shape[-1]
            self_decode_subquery_input_batch["labels"] = [
                [self.label_pad_token_id] * (self_decode_subquery_input_ids_length - len(label)) + label
                for label in self_decode_subquery_labels
            ]
            self_decode_subquery_input_batch["labels"] = torch.tensor(self_decode_subquery_input_batch["labels"], dtype=torch.int64)
            self_decode_subquery_input_batch["position_ids"] = [
                [0] * (self_decode_subquery_input_ids_length - len(position_id)) + position_id
                for position_id in self_decode_subquery_position_ids
            ]
            self_decode_subquery_input_batch["position_ids"] = torch.tensor(self_decode_subquery_input_batch["position_ids"], dtype=torch.int64)

        llm_input_batches = {
            'input_batch': input_batch,
            'self_decode_thought_input_batch': self_decode_thought_input_batch,
            'self_decode_subquery_input_batch': self_decode_subquery_input_batch,
        }
        retrieval_data = {
            'query_ids': all_query_ids, 
            'queries': all_queries,
            'thoughts': all_thoughts, 
            'subqueries': all_subqueries, 
            'pos_chunk_items': all_pos_chunk_items,
            'hard_neg_chunk_items': all_hard_neg_chunk_items
        }

        return llm_input_batches, retrieval_data

    def __getitem__(self, index):

        batch_indices = self.all_batch_indices[index]
        batch_data = [self.data[idx] for idx in batch_indices]
        batch_data = self.process_batch_data(batch_data)
        # batch_data = self.all_batch_data[index]\

        return batch_data

