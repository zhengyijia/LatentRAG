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
CUDA_VISIBLE_DEVICES="0,1,2" nohup bash launch_qwen3-embedding-0.6b.sh > launch_qwen3-embedding-0.6b.log 2>&1 &
SERVER_PID=$!
cd ../evaluation_with_step_wise_time

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

# run experiments
# # 7B
# echo "Running eval_few_shot_rag_qwen2.5-7b-instruct_qwen3-embedding-0.6b.sh"
# CUDA_VISIBLE_DEVICES="3" bash eval_few_shot_rag_qwen2.5-7b-instruct_qwen3-embedding-0.6b.sh > eval_few_shot_rag_qwen2.5-7b-instruct_qwen3-embedding-0.6b_4gpus.log 2>&1
# echo "Running eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b.sh"
# CUDA_VISIBLE_DEVICES="3" bash eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b.sh > eval_searchr1_qwen2.5-7b-em-ppo_qwen3-embedding-0.6b_4gpus.log 2>&1
# echo "Running eval_autorefine_steps_qwen2.5-7b_qwen3-embedding-0.6b.sh"
# CUDA_VISIBLE_DEVICES="3" bash eval_autorefine_steps_qwen2.5-7b_qwen3-embedding-0.6b.sh > eval_autorefine_steps_qwen2.5-7b_qwen3-embedding-0.6b_4gpus.log 2>&1
# echo "Running eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_Search-R1.sh"
# CUDA_VISIBLE_DEVICES="3" bash eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_Search-R1.sh > eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_Search-R1_4gpus.log 2>&1
# echo "Running eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_AutoRefine.sh"
# CUDA_VISIBLE_DEVICES="3" bash eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_AutoRefine.sh > eval_latentrag_qwen2.5-7b-base_qwen3-embedding-0.6b_AutoRefine_4gpus.log 2>&1

# # 3B
# echo "Running eval_few_shot_rag_qwen2.5-3b-instruct_qwen3-embedding-0.6b.sh"
# CUDA_VISIBLE_DEVICES="3" bash eval_few_shot_rag_qwen2.5-3b-instruct_qwen3-embedding-0.6b.sh > eval_few_shot_rag_qwen2.5-3b-instruct_qwen3-embedding-0.6b_4gpus.log 2>&1
# echo "Running eval_searchr1_qwen2.5-3b-em-ppo_qwen3-embedding-0.6b.sh"
# CUDA_VISIBLE_DEVICES="3" bash eval_searchr1_qwen2.5-3b-em-ppo_qwen3-embedding-0.6b.sh > eval_searchr1_qwen2.5-3b-em-ppo_qwen3-embedding-0.6b_4gpus.log 2>&1
# echo "Running eval_latentrag_qwen2.5-3b-base_qwen3-embedding-0.6b_Search-R1.sh"
# CUDA_VISIBLE_DEVICES="3" bash eval_latentrag_qwen2.5-3b-base_qwen3-embedding-0.6b_Search-R1.sh > eval_latentrag_qwen2.5-3b-base_qwen3-embedding-0.6b_Search-R1_4gpus.log 2>&1

# 14B
echo "Running eval_few_shot_rag_qwen2.5-14b-instruct_qwen3-embedding-0.6b.sh"
CUDA_VISIBLE_DEVICES="3" bash eval_few_shot_rag_qwen2.5-14b-instruct_qwen3-embedding-0.6b.sh > eval_few_shot_rag_qwen2.5-14b-instruct_qwen3-embedding-0.6b_4gpus.log 2>&1
echo "Running eval_searchr1_qwen2.5-14b-em-ppo_qwen3-embedding-0.6b.sh"
CUDA_VISIBLE_DEVICES="3" bash eval_searchr1_qwen2.5-14b-em-ppo_qwen3-embedding-0.6b.sh > eval_searchr1_qwen2.5-14b-em-ppo_qwen3-embedding-0.6b_4gpus.log 2>&1
echo "Running eval_latentrag_qwen2.5-14b-base_qwen3-embedding-0.6b_Search-R1.sh"
CUDA_VISIBLE_DEVICES="3" bash eval_latentrag_qwen2.5-14b-base_qwen3-embedding-0.6b_Search-R1.sh > eval_latentrag_qwen2.5-14b-base_qwen3-embedding-0.6b_Search-R1_4gpus.log 2>&1


