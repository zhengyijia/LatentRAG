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

# start retriever server
cd ../launch_retriever_server
CUDA_VISIBLE_DEVICES="0" nohup bash launch_qwen3-embedding-0.6b.sh > launch_qwen3-embedding-0.6b.log 2>&1 &
SERVER_PID=$!
cd ../evaluation

READY=0
for i in {1..60}; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: retriever process died unexpectedly" >&2
        exit 1
    fi

    if curl -s http://localhost:8007/retriever_info 2>/dev/null | grep -q "qwen3-embedding-0.6b"; then
        READY=1
        echo "Retriever server with qwen3-embedding-0.6b is ready!"
        break
    fi
    sleep 1m
done

# make sure the retriever server is ready before proceeding
if [ $READY -ne 1 ]; then
    echo "ERROR: Retriever server failed to start within the timeout period" >&2
    exit 1
fi

# # run experiments

# run vllm experiments with 7B models and 0.5B draft model
echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_2tokens_qwen0.5b.sh"
CUDA_VISIBLE_DEVICES="0" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_2tokens_qwen0.5b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_2tokens_qwen0.5b.log 2>&1
# echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_3tokens_qwen0.5b.sh"
# CUDA_VISIBLE_DEVICES="0" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_3tokens_qwen0.5b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_3tokens_qwen0.5b.log 2>&1
# echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_5tokens_qwen0.5b.sh"
# CUDA_VISIBLE_DEVICES="0" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_5tokens_qwen0.5b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_5tokens_qwen0.5b.log 2>&1
# echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_10tokens_qwen0.5b.sh"
# CUDA_VISIBLE_DEVICES="0" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_10tokens_qwen0.5b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_10tokens_qwen0.5b.log 2>&1

# run vllm experiments with 7B models and 0.5B sft draft model
echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_2tokens_sft_qwen0.5b.sh"
CUDA_VISIBLE_DEVICES="0" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_2tokens_sft_qwen0.5b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_2tokens_sft_qwen0.5b.log 2>&1
# echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_3tokens_sft_qwen0.5b.sh"
# CUDA_VISIBLE_DEVICES="0" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_3tokens_sft_qwen0.5b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_3tokens_sft_qwen0.5b.log 2>&1
# echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_5tokens_sft_qwen0.5b.sh"
# CUDA_VISIBLE_DEVICES="0" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_5tokens_sft_qwen0.5b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_5tokens_sft_qwen0.5b.log 2>&1
# echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_10tokens_sft_qwen0.5b.sh"
# CUDA_VISIBLE_DEVICES="0" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_10tokens_sft_qwen0.5b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_vllm_bs1_speculative_10tokens_sft_qwen0.5b.log 2>&1

