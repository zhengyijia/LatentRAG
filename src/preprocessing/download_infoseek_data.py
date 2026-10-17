import argparse
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import requests
import pandas as pd
from io import StringIO
import jsonlines

parser = argparse.ArgumentParser(description="Download InfoSeek QA data and the corpus.")
parser.add_argument("--save_path", type=str, default='data/deepresearch/infoseek', help="Local directory to save files")
args = parser.parse_args()

# download qa data
print("Downloading InfoSeek QA data...", flush=True)
url='https://raw.githubusercontent.com/Yuqi-Zhou/LRAT/4ea45eb650ade942765271b4c81664c2bd8c09c2/datasets/InfoSeek-Eval.tsv'
r = requests.get(url)
r.raise_for_status()
test_data = pd.read_csv(StringIO(r.text), sep="\t", header=None)
test_data.columns = ["id", "question", "golden_answer"]
test_data = test_data.to_dict(orient="records")
for item in test_data:
    item['golden_answers'] = [item.pop('golden_answer')] # follow the general format of other qa datasets

test_question_set = set(item['question'] for item in test_data)


train_ds = load_dataset("Lk123/InfoSeek", data_files={"train": "data/InfoSeekQA.jsonl"})["train"]
train_data = []
for item in train_ds:
    # make sure there is no overlap between train and test questions
    if item['question'] not in test_question_set:
        item['golden_answers'] = [item.pop('answer')] # follow the general format of other qa datasets
        train_data.append(item)
train_data = [{'id': f'InfoSeek-Train_{idx+1}', **item} for idx, item in enumerate(train_data)]

qa_save_dir = os.path.join(args.save_path, 'qa_data', 'infoseek')
os.makedirs(qa_save_dir, exist_ok=True)
train_save_path = os.path.join(qa_save_dir, 'train.jsonl')
with jsonlines.open(train_save_path, mode='w') as writer:
    writer.write_all(train_data)
test_save_path = os.path.join(qa_save_dir, 'test.jsonl')
with jsonlines.open(test_save_path, mode='w') as writer:
    writer.write_all(test_data)
    
# download corpus
print("Downloading corpus...", flush=True)
# corpus_ds = load_dataset("Lk123/wiki-25-512", data_files={"train": "wiki-25-512.jsonl"})["train"]
# corpus_data = [item for item in corpus_ds]
# corpus_save_dir = os.path.join(args.save_path, 'corpus')
# os.makedirs(corpus_save_dir, exist_ok=True)
# corpus_save_path = os.path.join(corpus_save_dir, 'wiki-25-512.jsonl')
# with jsonlines.open(corpus_save_path, mode='w') as writer:
#     writer.write_all(corpus_data)

corpus_save_dir = os.path.join(args.save_path, 'corpus')
hf_hub_download(
        repo_id="Lk123/wiki-25-512",
        filename="wiki-25-512.jsonl", 
        repo_type="dataset",
        local_dir=corpus_save_dir
)

print("Done.", flush=True)

