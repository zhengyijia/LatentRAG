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


GPU_ENV_CONFIG="default"

# EVAL_FILES="['nq/train.jsonl', 'hotpotqa/train.jsonl']"
EVAL_FILES="['nq/test.jsonl', 'triviaqa/test.jsonl', 'popqa/test.jsonl', 'hotpotqa/dev.jsonl', '2wikimultihopqa/dev.jsonl', 'musique/dev.jsonl', 'bamboogle/test.jsonl']"
# EVAL_FILES="['bamboogle/test.jsonl']"

############## MODEL PATH CONFIG ##############

# based on Search-R1
# qwen3-embedding-0.6b
# LORA_PATH="$(pwd)/trainer_output/SFTModel_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B/20260517_143915_lr0.0001_bs16_ep5_lora_r16_dp0.1_ls0.0"

# based on AutoRefine
# qwen3-embedding-0.6b
LORA_PATH="$(pwd)/trainer_output/SFTModel_Qwen2.5-7B_qwen3-embedding-0.6b_AutoRefine_Qwen2.5-7B/20260521_135340_lr0.0001_bs16_ep5_lora_r16_dp0.1_ls0.0"

################################################

python src/evaluate_with_step_wise_time.py \
  --config-path $LORA_PATH \
  --config-name training_config \
  hydra.run.dir=. hydra.output_subdir=null \
  "data.data_path='$GLOBAL_DATA_PATH/data/qa_data'" \
  data.eval_files="$EVAL_FILES" \
  +data.save_path="$LORA_PATH/evaluation_with_step_wise_time_output" \
  "model.lora.lora_name_or_path='$LORA_PATH'" \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/corpus'" \
  gpu_env="${GPU_ENV_CONFIG}" \
  training.overwrite_output_dir=false \
  "+data.sampling=100" \
  vllm.use_vllm=false
#   "+retriever.base_url='http://localhost:8007'"
