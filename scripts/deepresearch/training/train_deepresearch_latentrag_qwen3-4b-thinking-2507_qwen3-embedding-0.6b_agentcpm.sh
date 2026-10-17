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

MODEL_CONFIG="latentrag_qwen3-4b-thinking-2507"
RETRIEVER_CONFIG="qwen3-embedding-0.6b"
DATA_TAG="AgentCPM-4B"

# GPU_ENV_CONFIG="${MODEL_CONFIG}_${RETRIEVER_CONFIG}/gpu_${NUM_GPUS}_${MEM_GB_TIER}"
GPU_ENV_CONFIG="default"

# set global data path
if [ -n "$GLOBAL_DATA_PATH" ]; then
  project_name=$(basename "$(pwd)")
  GLOBAL_DATA_PATH="$GLOBAL_DATA_PATH/$project_name"
else
  GLOBAL_DATA_PATH="$(pwd)"
fi

export WANDB_PROJECT="latentrag"
RUN_NAME_suffix=""

TRAIN_FILES="['infoseek/train/${DATA_TAG}_${RETRIEVER_CONFIG}_maxturns_30_topk_3_negk_7_reasoning_steps_judged_llm_judge_correct.json']"
EVAL_FILES=null

python -m torch.distributed.run --rdzv-endpoint 127.0.0.1:12346 \
  --nnodes 1 --nproc_per_node ${NUM_GPUS} src/train_deepresearch.py \
  --config-path $(pwd)/config \
  --config-name ${MODEL_CONFIG} \
  "+data.data_tag='${DATA_TAG}'" \
  "+data.data_path='$GLOBAL_DATA_PATH/data/deepresearch/infoseek/reasoning_steps'" \
  +data.train_files="$TRAIN_FILES" \
  +data.eval_files="$EVAL_FILES" \
  "retriever=${RETRIEVER_CONFIG}" \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/deepresearch/infoseek/corpus'" \
  gpu_env="${GPU_ENV_CONFIG}" \
  training.overwrite_output_dir=true \
  custom_training.align_loss='mse_kl_rank_loss' \
  training.num_train_epochs=10 \
  custom_training.align_loss_weight=1.0 \
  custom_training.gen_loss_weight=1.0 \
  custom_training.self_decode_thought_gen_loss_weight=1.0 \
  custom_training.self_decode_subquery_gen_loss_weight=1.0 \
  custom_training.align_mse_loss_weight=10.0 \
  custom_training.align_nce_loss_weight=1.0 \
  custom_training.temperature=0.03 \
  custom_training.decode_thought=true \
  custom_training.decode_subquery=true \
  retriever.encoder_trainable=true \
  retriever.encoder_lora.use_lora=true \
  retriever.share_encoder=false \
  model.projector_type='self_attn' \
  model.num_latent_thought_tokens=16 \
  model.num_latent_subquery_tokens=16 \
  +training.run_name="$RUN_NAME_suffix" \
  data.filter_em_correct=false \
  data.filter_cover_em_correct=false \
  data.filter_llm_judge_correct=true \
  data.n_bins=1 \
  +data.title_in_first_line=false \
  custom_training.neg_k=7 \
  +custom_training.random_neg_k=8 \
  data.max_seq_length=20000 \
  data.min_seq_length=0 \
  +data.split_train_eval_set=true \
  +data.num_eval_samples=64 \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2
  # data.max_seq_length=130000 \
  # data.min_seq_length=0 \
  # +custom_training.random_neg_k=8 \
