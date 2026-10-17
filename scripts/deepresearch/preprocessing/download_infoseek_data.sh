#!/bin/bash
set -e

cd ../../..

# set up python environment
source scripts/common_settings.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

save_path=data/deepresearch/infoseek

python src/preprocessing/download_infoseek_data.py \
    --save_path "$save_path"

rm -rf $save_path/corpus/.cache
