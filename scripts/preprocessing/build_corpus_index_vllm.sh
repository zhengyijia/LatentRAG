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

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# NOTICE: vLLM (0.9.2) does not support max_model_len larger than `max_position_embeddings` in the retriever config
# https://github.com/vllm-project/vllm/pull/20837
# for e5-base-v2, the max_position_embeddings is 512 and some information will be truncated.
# We recommend to use transformer based implementation instead of this vLLM implementation for these models.

# build index - Qwen3-Embedding-0.6B
python src/preprocessing/build_index.py \
  --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
  --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_qwen3-embedding-0.6b_fp16_shard \
  --model_name qwen3-embedding-0.6b \
  --model_path Qwen/Qwen3-Embedding-0.6B \
  --encode_batch_size 128 \
  --shard_size 100000 \
  --use_vllm \
  --max_length 2048 \
  --save_fp16

# # build index - Qwen3-Embedding-4B
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_qwen3-embedding-4b_fp16_shard \
#   --model_name qwen3-embedding-4b \
#   --model_path Qwen/Qwen3-Embedding-4B \
#   --encode_batch_size 128 \
#   --shard_size 100000 \
#   --use_vllm \
#   --max_length 2048 \
#   --save_fp16 \
#   --shard_range_begin 95

# # build index - Qwen3-Embedding-8B
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w_qwen3-embedding-8b_fp16_shard \
#   --model_name qwen3-embedding-8b \
#   --model_path Qwen/Qwen3-Embedding-8B \
#   --encode_batch_size 240 \
#   --shard_size 100000 \
#   --use_vllm \
#   --max_length 2048 \
#   --save_fp16 \
#   --shard_range_begin 128
#   # --shard_range_end 9
