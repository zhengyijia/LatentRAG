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


GPU_ENV_CONFIG="default"

EVAL_FILES="['infoseek/test.jsonl']"

LORA_PATH="$(pwd)/trainer_output/deepresearch/LatentRAG_Qwen3-4B-Thinking-2507_qwen3-embedding-0.6b_AgentCPM-4B/20260710_212829_lr0.0001_bs4_ep10_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens16_subquery_tokens16/20260710_212829_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"


python src/evaluate_with_step_wise_time.py \
  --config-path $LORA_PATH \
  --config-name training_config \
  hydra.run.dir=. hydra.output_subdir=null \
  "data.data_path='$GLOBAL_DATA_PATH/data/deepresearch/infoseek/qa_data'" \
  data.eval_files="$EVAL_FILES" \
  +data.save_path="$LORA_PATH/evaluation_with_step_wise_time_output" \
  "model.lora.lora_name_or_path='$LORA_PATH'" \
  "model.projector_path='$LORA_PATH'" \
  "model.self_thought_projector_path='$LORA_PATH'" \
  "model.self_subquery_projector_path='$LORA_PATH'" \
  "+retriever.custom_encoder_name_or_path='${LORA_PATH}/encoder'" \
  "retriever.encoder_lora.lora_name_or_path='${LORA_PATH}/encoder'" \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/deepresearch/infoseek/corpus'" \
  "retriever.corpus_file='wiki-25-512.jsonl'" \
  "retriever.corpus_index_folder='wiki-25-512_qwen3-embedding-0.6b_fp16_shard'" \
  data.batch_size=1 \
  gpu_env="${GPU_ENV_CONFIG}" \
  training.overwrite_output_dir=false \
  +batch_subquery_decoding=false \
  data.filter_em_correct=false \
  data.filter_llm_judge_correct=false \
  model.max_turns=30 \
  "+data.sampling=50"
#   "+retriever.base_url='http://localhost:8007'"
