#!/bin/bash
set -e

cd ../..

# set up python environment
source scripts/common_settings.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

save_path=data/qa_data

datasets=(
    # train datasets
    "RUC-NLPIR/FlashRAG_datasets nq train"
    "RUC-NLPIR/FlashRAG_datasets hotpotqa train"

    # test datasets
    "RUC-NLPIR/FlashRAG_datasets nq test"
    "RUC-NLPIR/FlashRAG_datasets triviaqa test"
    "RUC-NLPIR/FlashRAG_datasets popqa test"
    "RUC-NLPIR/FlashRAG_datasets hotpotqa dev"
    "RUC-NLPIR/FlashRAG_datasets 2wikimultihopqa dev"
    "RUC-NLPIR/FlashRAG_datasets musique dev"
    "RUC-NLPIR/FlashRAG_datasets bamboogle test"
)

for dataset_config in "${datasets[@]}"; do
    echo "Downloading dataset: $dataset_config"
    read -r repo_id dataset_name dataset_split <<< "$dataset_config"
    python src/preprocessing/download_qa_data.py \
        --repo_id "$repo_id" \
        --dataset_name "$dataset_name" \
        --dataset_split "$dataset_split" \
        --save_path "$save_path"
done

rm -rf $save_path/.cache

