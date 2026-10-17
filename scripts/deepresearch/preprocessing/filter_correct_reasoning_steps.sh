#!/bin/bash
set -e

cd ../../..

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

INPUT_PATH="$GLOBAL_DATA_PATH/data/deepresearch/infoseek/reasoning_steps/infoseek/train/AgentCPM-4B_qwen3-embedding-0.6b_maxturns_30_topk_3_negk_7_reasoning_steps_judged.json"

python -u src/preprocessing/filter_reasoning_steps.py \
    --input_path "$INPUT_PATH" \
    --filter_llm_judge_correct

