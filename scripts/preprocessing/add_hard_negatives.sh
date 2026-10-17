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

# set global data path
if [ -n "$GLOBAL_DATA_PATH" ]; then
  project_name=$(basename "$(pwd)")
  GLOBAL_DATA_PATH="$GLOBAL_DATA_PATH/$project_name"
else
  GLOBAL_DATA_PATH="$(pwd)"
fi

python src/preprocessing/add_hard_negatives.py \
  --config-path "$(pwd)/trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B/20260426_185237_lr0.0001_bs16_ep5_lora_r16_dp0.1_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16_no_mse_loss/20260426_185237_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought" \
  --config-name training_config \
  hydra.run.dir=. hydra.output_subdir=null \
  "data.data_path='trainer_output/LatentRAG_Qwen2.5-7B_qwen3-embedding-0.6b_Search-R1_Qwen2.5-7B/20260426_185237_lr0.0001_bs16_ep5_lora_r16_dp0.1_kl_rank_loss_temp0.03_ls0.0_self_attn_encoder_trainable_encoder_lora_thought_tokens4_subquery_tokens16_no_mse_loss/20260426_185237_alw1.0_glw1.0_sdtw1.0_sdsw1.0_mlw10.0_nlw1.0_decode_subquery_decode_thought/evaluation_output'" \
  "data.eval_files=['EVAL_batch_subquery_decoding_bs1_wiki18_100w_Qwen2.5-7B-Instruct_pretrain_50000.jsonl']" \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/corpus'" \
  "+retriever.neg_k=15" \
  "+retriever.base_url='http://localhost:8007'"



