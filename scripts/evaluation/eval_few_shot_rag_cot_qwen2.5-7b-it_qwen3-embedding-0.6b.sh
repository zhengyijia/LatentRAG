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

# get number of gpus
NUM_GPUS=$(python - <<EOF
import torch
print(torch.cuda.device_count())
EOF
)
echo "NUM_GPUS: ${NUM_GPUS}"

# get gpu memory info of the first visible gpu device
MEM_GB=$(python - <<EOF
import torch
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(torch.cuda.get_device_properties(0).total_memory // (1024**3))
else:
    print(0)
EOF
)

if [ $MEM_GB -lt 60 ]; then
    MEM_GB_TIER="40g"
elif [ $MEM_GB -lt 90 ]; then
    MEM_GB_TIER="64g"
else
    MEM_GB_TIER="94g"
fi
echo "MEM_GB: ${MEM_GB}, MEM_GB_TIER: ${MEM_GB_TIER}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# set global data path
if [ -n "$GLOBAL_DATA_PATH" ]; then
  project_name=$(basename "$(pwd)")
  GLOBAL_DATA_PATH="$GLOBAL_DATA_PATH/$project_name"
else
  GLOBAL_DATA_PATH="$(pwd)"
fi

MODEL_CONFIG="few_shot_rag_cot_qwen2.5-7b-instruct"
RETRIEVER_CONFIG="qwen3-embedding-0.6b"

# GPU_ENV_CONFIG="${MODEL_CONFIG}_${RETRIEVER_CONFIG}/gpu_${NUM_GPUS}_${MEM_GB_TIER}"
GPU_ENV_CONFIG="default"

# EVAL_FILES="['nq/train.jsonl', 'hotpotqa/train.jsonl']"
EVAL_FILES="['nq/test.jsonl', 'triviaqa/test.jsonl', 'popqa/test.jsonl', 'hotpotqa/dev.jsonl', '2wikimultihopqa/dev.jsonl', 'musique/dev.jsonl', 'bamboogle/test.jsonl']"
# EVAL_FILES="['bamboogle/test.jsonl']"

python src/evaluate.py \
  --config-path $(pwd)/config \
  --config-name ${MODEL_CONFIG} \
  hydra.run.dir=. hydra.output_subdir=null \
  "+data.data_path='$GLOBAL_DATA_PATH/data/qa_data'" \
  +data.eval_files="$EVAL_FILES" \
  +data.save_path="$(pwd)/trainer_output/${MODEL_CONFIG}_${RETRIEVER_CONFIG}/evaluation_output" \
  retriever=${RETRIEVER_CONFIG} \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/corpus'" \
  gpu_env="${GPU_ENV_CONFIG}" \
  training.overwrite_output_dir=false \
  "+data.sampling=100"
