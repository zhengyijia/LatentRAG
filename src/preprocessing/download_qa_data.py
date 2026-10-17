import os
import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download datasets from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default=None, help="Hugging Face repository ID")
parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
parser.add_argument("--dataset_split", type=str, default=None, help="Split of the dataset to download")
parser.add_argument("--save_path", type=str, default='data/qa_data', help="Local directory to save files")

args = parser.parse_args()

hf_hub_download(
    repo_id=args.repo_id,
    filename=os.path.join(args.dataset_name, args.dataset_split + '.jsonl'),
    repo_type="dataset",
    local_dir=args.save_path,
)
