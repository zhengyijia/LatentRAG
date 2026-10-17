#!/bin/bash
set -e

cd ../..

# set up python environment
source scripts/common_settings.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

save_path=data/corpus
python src/preprocessing/download_corpus.py --save_path $save_path
rm -rf $save_path/.cache

# extract corpus
unzip $save_path/retrieval-corpus/wiki18_100w.zip -d $save_path && rm -rf $save_path/retrieval-corpus

