import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default="RUC-NLPIR/FlashRAG_datasets", help="Hugging Face repository ID")
parser.add_argument("--filename", type=str, default="retrieval-corpus/wiki18_100w.zip", help="File to download from the repository")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
    
args = parser.parse_args()

hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type="dataset",
        local_dir=args.save_path,
)
