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
# e5-base-v2
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_e5-base-v2_Search-R1_Qwen2.5-7B/20260417_212958_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.01_ls0.0_self_attn_encoder_trainable_thought_tokens4_subquery_tokens16/20260417_212958_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# qwen3-embedding-0.6b
LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B/20260417_213416_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16/20260417_213416_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# jina-embeddings-v5-text-nano
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_jina-embeddings-v5-text-nano_Search-R1_Qwen2.5-7B/20260331_230349_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_thought_tokens4_subquery_tokens16/20260331_230349_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# harrier-oss-v1-270m
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_harrier-oss-v1-270m_Search-R1_Qwen2.5-7B/20260402_171048_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_thought_tokens4_subquery_tokens16/20260402_171048_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# F2LLM-v2-330M
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_F2LLM-v2-330M_Search-R1_Qwen2.5-7B/20260405_150002_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_thought_tokens4_subquery_tokens16/20260405_150002_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# qwen3-embedding-4b
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-4b_Search-R1_Qwen2.5-7B/20260421_160817_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16_lora_r16/20260421_160817_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# qwen3-embedding-8b
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-8b_Search-R1_Qwen2.5-7B/20260423_205129_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16/20260423_205129_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"


# based on AutoRefine
# e5-base-v2
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_e5-base-v2_AutoRefine_Qwen2.5-7B/20260414_214603_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.01_ls0.0_self_attn_encoder_trainable_thought_tokens4_subquery_tokens16/20260414_214603_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# qwen3-embedding-0.6b
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_AutoRefine_Qwen2.5-7B/20260409_112148_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16/20260409_112148_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# jina-embeddings-v5-text-nano
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_jina-embeddings-v5-text-nano_AutoRefine_Qwen2.5-7B/20260409_112646_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_thought_tokens4_subquery_tokens16/20260409_112646_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# harrier-oss-v1-270m
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_harrier-oss-v1-270m_AutoRefine_Qwen2.5-7B/20260413_092208_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_thought_tokens4_subquery_tokens16/20260413_092208_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# F2LLM-v2-330M
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_F2LLM-v2-330M_AutoRefine_Qwen2.5-7B/20260413_173234_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_thought_tokens4_subquery_tokens16/20260413_173234_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# qwen3-embedding-4b
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-4b_AutoRefine_Qwen2.5-7B/20260425_092321_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16/20260425_092321_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"
# qwen3-embedding-8b
# LORA_PATH="$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-8b_AutoRefine_Qwen2.5-7B/20260425_092736_lr0.0001_bs16_ep5_lora_r16_dp0.1_mse_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16/20260425_092736_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought"


################################################

python src/evaluate_with_step_wise_time.py \
  --config-path $LORA_PATH \
  --config-name training_config \
  hydra.run.dir=. hydra.output_subdir=null \
  "data.data_path='$GLOBAL_DATA_PATH/data/qa_data'" \
  data.eval_files="$EVAL_FILES" \
  +data.save_path="$LORA_PATH/evaluation_with_step_wise_time_output" \
  "model.lora.lora_name_or_path='$LORA_PATH'" \
  "model.projector_path='$LORA_PATH'" \
  "model.self_thought_projector_path='$LORA_PATH'" \
  "model.self_subquery_projector_path='$LORA_PATH'" \
  "+retriever.custom_encoder_name_or_path='${LORA_PATH}/encoder'" \
  "retriever.encoder_lora.lora_name_or_path='${LORA_PATH}/encoder'" \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/corpus'" \
  gpu_env="${GPU_ENV_CONFIG}" \
  training.overwrite_output_dir=false \
  +batch_subquery_decoding=false \
  data.filter_em_correct=false \
  data.filter_llm_judge_correct=false \
  "+data.sampling=100" \
  "+retriever.base_url='http://localhost:8007'"
