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


MODEL_CONFIG="agentcpm-4b"
RETRIEVER_CONFIG="qwen3-embedding-0.6b"

GPU_ENV_CONFIG="default"

EVAL_FILES="['infoseek/test.jsonl']"

python src/evaluate.py \
  --config-path $(pwd)/config \
  --config-name ${MODEL_CONFIG} \
  hydra.run.dir=. hydra.output_subdir=null \
  "+data.data_path='$GLOBAL_DATA_PATH/data/deepresearch/infoseek/qa_data'" \
  +data.eval_files="$EVAL_FILES" \
  +data.save_path="$(pwd)/trainer_output/deepresearch/${MODEL_CONFIG}_${RETRIEVER_CONFIG}/evaluation_output_vllm_bs1" \
  retriever=${RETRIEVER_CONFIG} \
  "retriever.corpus_path='$GLOBAL_DATA_PATH/data/deepresearch/infoseek/corpus'" \
  "retriever.corpus_file='wiki-25-512.jsonl'" \
  "retriever.corpus_index_folder='wiki-25-512_${RETRIEVER_CONFIG}_fp16_shard'" \
  data.batch_size=1 \
  gpu_env="${GPU_ENV_CONFIG}" \
  training.overwrite_output_dir=false \
  +data.filter_em_correct=false \
  +data.filter_llm_judge_correct=false \
  model.max_turns=30 \
  vllm.use_vllm=true \
  vllm.qa_model_gpu_memory_utilization=0.6 \
  vllm.qa_model_max_model_len=130000 \
  vllm.qa_model_max_num_seqs=1 \
  "+data.sampling=50"
#   "+retriever.base_url='http://localhost:8007'"
