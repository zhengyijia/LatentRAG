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

REFERENCE_MODEL="AgentCPM-4B"
RETRIEVER="qwen3-embedding-0.6b"

EVAL_FILES="['browsecompplus/test.jsonl']"

LORA_PATH="$(pwd)/trainer_output/deepresearch/SFTModel_Qwen3-4B-Thinking-2507_qwen3-embedding-0.6b_AgentCPM-4B/20260710_214231_lr0.0001_bs4_ep10_lora_r16_dp0.1_ls0.0"

python src/evaluate.py \
  --config-path $LORA_PATH \
  --config-name training_config \
  hydra.run.dir=. hydra.output_subdir=null \
  "data.data_path='$GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/qa_data'" \
  data.eval_files="$EVAL_FILES" \
  +data.save_path="$LORA_PATH/evaluation_output_vllm_bs1" \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/corpus'" \
  "retriever.corpus_file='browsecompplus-corpus-512.jsonl'" \
  "retriever.corpus_index_folder='browsecompplus-corpus-512_qwen3-embedding-0.6b_fp16_shard'" \
  data.batch_size=1 \
  training.overwrite_output_dir=false \
  data.filter_em_correct=false \
  data.filter_llm_judge_correct=false \
  model.max_turns=30 \
  vllm.use_vllm=true \
  +vllm.qa_model_gpu_memory_utilization=0.85 \
  +vllm.qa_model_max_model_len=130000 \
  +vllm.qa_model_max_num_seqs=1 \
  "model.model_name_or_path='${LORA_PATH}/merged_model'" \
  "+data.sampling=50"
#   "+retriever.base_url='http://ivi-cn004:8007'"
  # +data.sampling=1000
