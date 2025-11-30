#!/bin/bash


DATASET="/root/baidupan/rice_pretrain_data_BGI_ST/all_counts"
LOG_INTERVAL=100
MAX_LENGTH=1200
per_proc_batch_size=32
vocab_path="/root/data/vocab.json"




python pretrain_single_gpu.py \
    --data-source $DATASET \
    --save-dir /root/save/eval-$(date +%b%d-%H-%M-%Y) \
    --vocab_path $vocab_path \
    --max-seq-len $MAX_LENGTH \
    --batch-size $per_proc_batch_size \
    --eval-batch-size $per_proc_batch_size \
    --epochs 1 \
    --log-interval $LOG_INTERVAL \
    --trunc-by-sample \
    --no-cls \
    --no-cce \
    --fp16
