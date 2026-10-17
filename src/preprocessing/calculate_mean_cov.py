import os
import glob
import re
import numpy as np
from tqdm import tqdm

def estimate_mean_cov(shard_paths, eps=1e-6):
    count = 0
    mean = None
    second_moment = None
    for path in tqdm(shard_paths, desc="Estimating mean and covariance"):
        embeddings = np.load(path).astype(np.float64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if mean is None or second_moment is None:
            d = embeddings.shape[1]
            mean = np.zeros(d, dtype=np.float64)
            second_moment = np.zeros((d, d), dtype=np.float64)
        
        count += embeddings.shape[0]
        mean += embeddings.sum(axis=0)
        second_moment += embeddings.T @ embeddings
        
    mean /= count
    cov = (second_moment - count * np.outer(mean, mean)) / (count - 1)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    
    return mean[None, :], cov, W


index_folder = "data/corpus/wiki18_100w_e5_fp32_shard"
save_file = "data/corpus/wiki18_100w_e5_fp32_mean_cov.npz"
# index_folder = "data/corpus/wiki18_100w_qwen3_0.6b_fp32_shard"
# save_file = "data/corpus/wiki18_100w_qwen3_0.6b_fp32_mean_cov.npz"

all_paths = glob.glob(os.path.join(index_folder, "shard_index_*.npy"))
shard_paths = [p for p in all_paths if re.search(r"shard_index_\d{4}\.npy$", p)]
def extract_number(path):
    match = re.search(r"shard_index_(\d{4})\.npy$", path)
    return int(match.group(1)) if match else -1
shard_paths.sort(key=extract_number)

mean, cov, W = estimate_mean_cov(shard_paths)
print("mean:", mean)
print("cov shape:", cov.shape)
print("W shape:", W.shape)
np.savez(save_file, mean=mean, cov=cov, W=W)