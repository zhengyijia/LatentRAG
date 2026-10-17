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
cd ../../..

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
  --file_path $GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/corpus/browsecompplus-corpus-512.jsonl \
  --save_path $GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/corpus/browsecompplus-corpus-512_qwen3-embedding-0.6b_fp16_shard \
  --model_name qwen3-embedding-0.6b \
  --model_path Qwen/Qwen3-Embedding-0.6B \
  --encode_batch_size 128 \
  --shard_size 25000 \
  --save_fp16

# # build index - Qwen3-Embedding-4B
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/corpus/browsecompplus-corpus-512.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/corpus/browsecompplus-corpus-512_qwen3-embedding-4b_fp16_shard \
#   --model_name qwen3-embedding-4b \
#   --model_path Qwen/Qwen3-Embedding-4B \
#   --encode_batch_size 128 \
#   --shard_size 25000 \
#   --save_fp16

# # build index - Qwen3-Embedding-8B
# python src/preprocessing/build_index.py \
#   --file_path $GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/corpus/browsecompplus-corpus-512.jsonl \
#   --save_path $GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/corpus/browsecompplus-corpus-512_qwen3-embedding-8b_fp16_shard \
#   --model_name qwen3-embedding-8b \
#   --model_path Qwen/Qwen3-Embedding-8B \
#   --encode_batch_size 128 \
#   --shard_size 25000 \
#   --save_fp16
