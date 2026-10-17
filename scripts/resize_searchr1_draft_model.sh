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
cd ..

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

MODEL_TAG="Qwen2.5-0.5B"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B"
# REFERENCE_MODEL_TAG="Search-R1_Qwen2.5-7B"
# REFERENCE_MODEL_NAME_OR_PATH="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
REFERENCE_MODEL_TAG="Search-R1_Qwen2.5-3B"
REFERENCE_MODEL_NAME_OR_PATH="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo"

python src/resize_searchr1_draft_model.py \
  --model_tag "$MODEL_TAG" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --reference_model_tag "$REFERENCE_MODEL_TAG" \
  --reference_model_name_or_path "$REFERENCE_MODEL_NAME_OR_PATH" \
  --save_path "trainer_output/resized_draft_model_${MODEL_TAG}_to_${REFERENCE_MODEL_TAG}"

