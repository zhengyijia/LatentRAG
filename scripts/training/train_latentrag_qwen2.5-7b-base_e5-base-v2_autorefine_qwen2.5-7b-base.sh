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
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# set up python environment
source scripts/common_settings.sh

# if CUDA_VISIBLE_DEVICES is not set, use all GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    export CUDA_VISIBLE_DEVICES
else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr -d ' ' | awk -F',' '{print NF}')
fi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "NUM_GPUS: ${NUM_GPUS}"

# # get gpu memory info of the first visible gpu device
# FIRST_VISIBLE_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1 | tr -d ' ')
# MEM_MiB=$(nvidia-smi -i "$FIRST_VISIBLE_GPU" --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
# MEM_GB=$((MEM_MiB / 1024))
# if [ $MEM_GB -lt 30 ]; then
#     MEM_GB_TIER="24g"
# elif [ $MEM_GB -lt 90 ]; then
#     MEM_GB_TIER="40g"
# else
#     MEM_GB_TIER="94g"
# fi
# echo "MEM_GB: ${MEM_GB}, MEM_GB_TIER: ${MEM_GB_TIER}"

MODEL_CONFIG="latentrag_qwen2.5-7b-base"
RETRIEVER_CONFIG="e5-base-v2"
DATA_TAG="AutoRefine_Qwen2.5-7B"

# GPU_ENV_CONFIG="${MODEL_CONFIG}_${RETRIEVER_CONFIG}/gpu_${NUM_GPUS}_${MEM_GB_TIER}"
GPU_ENV_CONFIG="${MODEL_CONFIG}_${RETRIEVER_CONFIG}/gpu_2_94g"

# set global data path
if [ -n "$GLOBAL_DATA_PATH" ]; then
  project_name=$(basename "$(pwd)")
  GLOBAL_DATA_PATH="$GLOBAL_DATA_PATH/$project_name"
else
  GLOBAL_DATA_PATH="$(pwd)"
fi

export WANDB_PROJECT="latentrag"
RUN_NAME_suffix=""

TRAIN_FILES="['nq/train/${DATA_TAG}_${RETRIEVER_CONFIG}_maxturns_4_topk_3_negk_15_reasoning_steps.json', 'hotpotqa/train/${DATA_TAG}_${RETRIEVER_CONFIG}_maxturns_4_topk_3_negk_15_reasoning_steps.json']"
# EVAL_FILES="['2wikimultihopqa/dev/${DATA_TAG}_${RETRIEVER_CONFIG}_negk_15_reasoning_steps.json']"
EVAL_FILES=null

torchrun --rdzv-endpoint 127.0.0.1:12347 \
  --nnodes 1 --nproc_per_node ${NUM_GPUS} src/train.py \
  --config-path $(pwd)/config \
  --config-name ${MODEL_CONFIG} \
  "+data.data_tag='${DATA_TAG}'" \
  "+data.data_path='$GLOBAL_DATA_PATH/data/reasoning_steps'" \
  +data.train_files="$TRAIN_FILES" \
  +data.eval_files="$EVAL_FILES" \
  "retriever=${RETRIEVER_CONFIG}" \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/corpus'" \
  gpu_env="${GPU_ENV_CONFIG}" \
  training.overwrite_output_dir=true \
  custom_training.align_loss='mse_kl_rank_loss' \
  training.num_train_epochs=5 \
  custom_training.align_loss_weight=1.0 \
  custom_training.gen_loss_weight=1.0 \
  custom_training.self_decode_thought_gen_loss_weight=1.0 \
  custom_training.self_decode_subquery_gen_loss_weight=1.0 \
  custom_training.align_mse_loss_weight=10.0 \
  custom_training.align_nce_loss_weight=1.0 \
  custom_training.temperature=0.01 \
  custom_training.decode_thought=true \
  custom_training.decode_subquery=true \
  retriever.encoder_trainable=true \
  retriever.encoder_lora.use_lora=false \
  retriever.share_encoder=false \
  model.projector_type='self_attn' \
  model.self_thought_projector_type='self_attn' \
  model.self_subquery_projector_type='self_attn' \
  model.num_latent_thought_tokens=4 \
  model.num_latent_subquery_tokens=16 \
  +training.run_name="$RUN_NAME_suffix" \
  +data.split_train_eval_set=true \
  +data.num_eval_samples=256

