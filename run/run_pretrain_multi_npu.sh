#!/bin/bash

# Load environment if needed (adjust according to local setup)
# export PATH=/usr/local/cuda-11.8/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Simplified NCCL Config for local multi-GPU training
# export OMP_NUM_THREADS=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Training config
DATASET="/root/rice_pretrain_data_BGI_ST/all_counts"
MAX_LENGTH=1200
per_proc_batch_size=64
epoches=20
nlayers=6
nheads=4
embsize=256
d_hid=256
log_interval=500
save_interval=3000
vocab_path="/root/data/vocab.json"
code_path="/root/riceFM/run"

cd $code_path

# 设置NPU环境变量
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=1
# export TASK_QUEUE_ENABLE=1
# export PTCOPY_ENABLE=1
# export COMBINED_ENABLE=1
# export HCCL_WHITELIST_DISABLE=1

# 获取主节点IP地址
MASTER_ADDR=$(hostname -I | awk '{print $1}')


# Run torchrun for 8 GPUs locally
# torchrun --nproc_per_node=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    pretrain_multi_npu.py \
    --data-source $DATASET \
    --save-dir /root/riceFM_save/eval-$(date +%b%d-%H-%M-%Y) \
    --nlayers $nlayers \
    --nheads $nheads \
    --d-hid $d_hid \
    --embsize $embsize \
    --log-interval $log_interval \
    --save-interval $save_interval \
    --vocab_path $vocab_path \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size 64 \
    --epochs $epoches \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16 