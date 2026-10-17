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

# INPUT_PATH="$GLOBAL_DATA_PATH/data/deepresearch/infoseek/reasoning_steps/infoseek/test/AgentCPM-4B_qwen3-embedding-0.6b_maxturns_30_topk_3_negk_7_reasoning_steps.json"
# INPUT_PATH="$GLOBAL_DATA_PATH/data/deepresearch/infoseek/reasoning_steps/infoseek/train/AgentCPM-4B_qwen3-embedding-0.6b_maxturns_30_topk_3_negk_7_reasoning_steps.json"
# INPUT_PATH="$GLOBAL_DATA_PATH/data/deepresearch/browsecompplus/reasoning_steps/browsecompplus/test/AgentCPM-4B_qwen3-embedding-0.6b_maxturns_30_topk_3_negk_7_reasoning_steps.json"
# INPUT_PATH="$(pwd)/trainer_output/deepresearch/SFTModel_Qwen3-4B-Thinking-2507_qwen3-embedding-0.6b_AgentCPM-4B/20260710_214231_lr0.0001_bs4_ep10_lora_r16_dp0.1_ls0.0/evaluation_based_on_reasoning_steps_output/EVAL_bs4_use_vllm_infoseek_test_AgentCPM-4B_qwen3-embedding-0.6b_maxturns_30_topk_3_negk_7_reasoning_steps_judged.json"
# INPUT_PATH="$(pwd)/trainer_output/deepresearch/LatentRAG_Qwen3-4B-Thinking-2507_qwen3-embedding-0.6b_AgentCPM-4B/20260710_212829_lr0.0001_bs4_ep10_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens16_subquery_tokens16/20260710_212829_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought/evaluation_based_on_reasoning_steps_output/EVAL_bs1_infoseek_test_AgentCPM-4B_qwen3-embedding-0.6b_maxturns_30_topk_3_negk_7_reasoning_steps_judged.json"

# INPUT_PATH="$(pwd)/trainer_output/deepresearch/LatentRAG_Qwen3-4B-Thinking-2507_qwen3-embedding-0.6b_AgentCPM-4B/20260710_212829_lr0.0001_bs4_ep10_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens16_subquery_tokens16/20260710_212829_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought/evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_infoseek_test_AgentCPM-4B_qwen3-embedding-0.6b_maxturns_30_topk_3_negk_7_reasoning_steps_judged.json"
INPUT_PATH="$(pwd)/trainer_output/deepresearch/LatentRAG_Qwen3-4B-Thinking-2507_qwen3-embedding-0.6b_AgentCPM-4B/20260710_212829_lr0.0001_bs4_ep10_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens16_subquery_tokens16/20260710_212829_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought/evaluation_based_on_reasoning_steps_output/EVAL_batch_subquery_decoding_bs1_browsecompplus_test_AgentCPM-4B_qwen3-embedding-0.6b_maxturns_30_topk_3_negk_7_reasoning_steps_judged.json"

python -u src/llm_judge.py \
    --input_path "$INPUT_PATH" \
    --model_name_or_path "Qwen/Qwen3-32B" \
    --max_num_seqs 32

