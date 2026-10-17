#!/bin/bash
set -e

cd ../..

# set up python environment
source scripts/common_settings.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

data_path=data/qa_data
data_files=(
    "boolq/dev.jsonl"
)

for data_file in "${data_files[@]}"; do
    echo "Converting boolean answers to string in file: $data_file"
    python src/preprocessing/convert_bool_ans_to_str.py \
        --data_path "$data_path" \
        --data_file "$data_file"
done


