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

RETRIEVER="qwen3-embedding-0.6b"

# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B/20260311_234850_lr0.0001_bs16_negbs0_ep5_lora_r16_dp0.1_mse_kl_rank_loss_teacher_temp0.03_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16/20260311_234850_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B/20260426_185237_lr0.0001_bs16_ep5_lora_r16_dp0.1_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16_no_mse_loss/20260426_185237_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"

EVAL_FILES="['evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_nq_test_Search-R1_Qwen2.5-7B_qwen3-embedding-0.6b_maxturns_4_topk_3_negk_15_reasoning_steps.json', 'evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_triviaqa_test_Search-R1_Qwen2.5-7B_qwen3-embedding-0.6b_maxturns_4_topk_3_negk_15_reasoning_steps.json', 'evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_popqa_test_Search-R1_Qwen2.5-7B_qwen3-embedding-0.6b_maxturns_4_topk_3_negk_15_reasoning_steps.json', 'evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_hotpotqa_dev_Search-R1_Qwen2.5-7B_qwen3-embedding-0.6b_maxturns_4_topk_3_negk_15_reasoning_steps.json', 'evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_2wikimultihopqa_dev_Search-R1_Qwen2.5-7B_qwen3-embedding-0.6b_maxturns_4_topk_3_negk_15_reasoning_steps.json', 'evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_musique_dev_Search-R1_Qwen2.5-7B_qwen3-embedding-0.6b_maxturns_4_topk_3_negk_15_reasoning_steps.json', 'evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_bamboogle_test_Search-R1_Qwen2.5-7B_qwen3-embedding-0.6b_maxturns_4_topk_3_negk_15_reasoning_steps.json']"

python src/gen_latent_logits.py \
  --config-path $LORA_PATH \
  --config-name training_config \
  hydra.run.dir=. hydra.output_subdir=null \
  "data.data_path='$LORA_PATH'" \
  data.eval_files="$EVAL_FILES" \
  +data.save_path="$LORA_PATH/latent_logits_output" \
  "model.lora.lora_name_or_path='$LORA_PATH'" \
  "model.projector_path='$LORA_PATH'" \
  "model.self_thought_projector_path='$LORA_PATH'" \
  "model.self_subquery_projector_path='$LORA_PATH'" \
  "retriever.custom_encoder_name_or_path='${LORA_PATH}/encoder'" \
  "retriever.encoder_lora.lora_name_or_path='${LORA_PATH}/encoder'" \
  training.overwrite_output_dir=false
#   +data.filter_em_correct=true \
#   +data.filter_em_incorrect=true \
#   +data.sampling=100
