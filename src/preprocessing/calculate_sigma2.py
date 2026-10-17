import os
import glob
import re
import numpy as np
from tqdm import tqdm

def estimate_mean_cov(shard_paths, eps=1e-6):
    count = 0
    mean = None
    for path in tqdm(shard_paths, desc="Estimating mean"):
        embeddings = np.load(path).astype(np.float64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if mean is None:
            d = embeddings.shape[1]
            mean = np.zeros(d, dtype=np.float64)
        
        count += embeddings.shape[0]
        mean += embeddings.sum(axis=0)
        
    mean /= count
    var_sum = 0
    for path in tqdm(shard_paths, desc="Estimating variance"):
        embeddings = np.load(path).astype(np.float64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        var_sum += ((embeddings - mean) ** 2).sum()
    sigma2 = var_sum / (count - 1)
    
    return mean[None, :], sigma2


index_folder = "data/corpus/wiki18_100w_e5_fp32_shard"
# save_file = "data/corpus/wiki18_100w_e5_fp32_sigma2.npz"
# index_folder = "data/corpus/wiki18_100w_qwen3_0.6b_fp32_shard"
# save_file = "data/corpus/wiki18_100w_qwen3_0.6b_fp32_sigma2.npz"


all_paths = glob.glob(os.path.join(index_folder, "shard_index_*.npy"))
shard_paths = [p for p in all_paths if re.search(r"shard_index_\d{4}\.npy$", p)]
def extract_number(path):
    match = re.search(r"shard_index_(\d{4})\.npy$", path)
    return int(match.group(1)) if match else -1
shard_paths.sort(key=extract_number)

mean, sigma2 = estimate_mean_cov(shard_paths)
print("mean:", mean)
print("sigma2:", sigma2)
# np.savez(save_file, mean=mean, sigma2=sigma2)