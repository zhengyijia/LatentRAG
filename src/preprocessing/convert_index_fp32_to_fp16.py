import os
import glob
import re
import numpy as np
from tqdm import tqdm
import gc

# load_index_folder = "data/corpus/wiki18_100w_e5-base-v2_fp32_shard"
# save_index_folder = "data/corpus/wiki18_100w_e5-base-v2_fp16_shard"
load_index_folder = "data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp32_shard"
save_index_folder = "data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard"
# load_index_folder = "data/corpus/wiki18_100w_qwen3-embedding-4b_fp32_shard"
# save_index_folder = "data/corpus/wiki18_100w_qwen3-embedding-4b_fp16_shard"
# load_index_folder = "data/corpus/wiki18_100w_qwen3-embedding-8b_fp32_shard"
# save_index_folder = "data/corpus/wiki18_100w_qwen3-embedding-8b_fp16_shard"

all_paths = glob.glob(os.path.join(load_index_folder, "shard_index_*.npy"))
shard_paths = [p for p in all_paths if re.search(r"shard_index_\d{4}\.npy$", p)]
def extract_number(path):
    match = re.search(r"shard_index_(\d{4})\.npy$", path)
    return int(match.group(1)) if match else -1
shard_paths.sort(key=extract_number)

assert len(shard_paths) > 0, f"No shard files found in {load_index_folder}"
os.makedirs(save_index_folder, exist_ok=True)

for path in tqdm(shard_paths, desc="Converting FP32 to FP16"):
    save_path = os.path.join(save_index_folder, os.path.basename(path))
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping conversion for this shard.")
        continue
    
    embeddings = np.load(path)
    embeddings_fp16 = embeddings.astype(np.float16)
    np.save(save_path, embeddings_fp16)
    
    del embeddings
    del embeddings_fp16
    gc.collect()
    
print("Conversion completed.")

