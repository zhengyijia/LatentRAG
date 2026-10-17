#!/bin/bash
set -e

# print slurm job info
echo "==============================="
echo "Slurm job info:"
echo "Job ID: $SLURM_JOB_ID"
echo "Node name: $SLURMD_NODENAME"
echo "Node list: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "==============================="

# change to project root directory
cd ../..

# set up python environment
source scripts/common_settings.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# set global data path
if [ -n "$GLOBAL_DATA_PATH" ]; then
  project_name=$(basename "$(pwd)")
  GLOBAL_DATA_PATH="$GLOBAL_DATA_PATH/$project_name"
else
  GLOBAL_DATA_PATH="$(pwd)"
fi

# build index - Qwen3-Embedding-0.6B
python src/preprocessing/build_index.py \
  --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
  --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard \
  --model_name qwen3-embedding-0.6b \
  --model_path Qwen/Qwen3-Embedding-0.6B \
  --encode_batch_size 128 \
  --shard_size 100000 \
  --save_fp16

# # build index - Qwen3-Embedding-4B
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_qwen3-embedding-4b_fp16_shard \
#   --model_name qwen3-embedding-4b \
#   --model_path Qwen/Qwen3-Embedding-4B \
#   --encode_batch_size 128 \
#   --shard_size 100000 \
#   --save_fp16

# # build index - Qwen3-Embedding-8B
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_qwen3-embedding-8b_fp16_shard \
#   --model_name qwen3-embedding-8b \
#   --model_path Qwen/Qwen3-Embedding-8B \
#   --encode_batch_size 128 \
#   --shard_size 100000 \
#   --save_fp16

# # build index - e5-base-v2
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_e5-base-v2_fp16_shard \
#   --model_name e5-base-v2 \
#   --model_path intfloat/e5-base-v2 \
#   --encode_batch_size 4096 \
#   --shard_size 100000 \
#   --save_fp16

# # build index - jina-embeddings-v5-text-nano
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_jina-embeddings-v5-text-nano_fp16_shard \
#   --model_name jina-embeddings-v5-text-nano \
#   --model_path jinaai/jina-embeddings-v5-text-nano \
#   --encode_batch_size 1024 \
#   --shard_size 100000 \
#   --save_fp16
#   # --shard_range_end 100
#   # --shard_range_begin 100

# # build index - harrier-oss-v1-270m
# # harrier-oss-v1-270m seems have some issues with batch encoding, sometimes it will produce same embedding for the whole batch
# CUDA_VISIBLE_DEVICES="0" python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_harrier-oss-v1-270m_fp16_shard \
#   --model_name harrier-oss-v1-270m \
#   --model_path microsoft/harrier-oss-v1-270m \
#   --encode_batch_size 1 \
#   --shard_size 100000 \
#   --save_fp16 \
#   --shard_range_begin 150
#   # --encode_batch_size 1024 \
#   # --shard_range_end 100
#   # --shard_range_begin 100

# # build index - codefuse-ai/F2LLM-v2-330M
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_F2LLM-v2-330M_fp16_shard \
#   --model_name F2LLM-v2-330M \
#   --model_path codefuse-ai/F2LLM-v2-330M \
#   --encode_batch_size 1024 \
#   --shard_size 100000 \
#   --save_fp16 \
#   --shard_range_begin 200
#   # --shard_range_end 100
