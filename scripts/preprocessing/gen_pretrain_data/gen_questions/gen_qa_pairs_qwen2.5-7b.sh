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
cd ../../../..

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

MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
NUM_SAMPLES=50000

# build index - Qwen3-Embedding-4B
python src/gen_pretrain_data/gen_qa_pairs.py \
  --file_path $GLOBAL_DATA_PATH/data/corpus/wiki18_100w.jsonl \
  --save_path "$GLOBAL_DATA_PATH/data/pretrain_data/wiki18_100w_${MODEL_NAME}/pretrain_${NUM_SAMPLES}.jsonl" \
  --model_name $MODEL_NAME \
  --model_path $MODEL_PATH \
  --max_length 2048 \
  --max_num_seqs 512 \
  --batch_size 100000 \
  --num_samples $NUM_SAMPLES \
  --use_chat_template
