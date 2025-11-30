#!/bin/bash

data_path="/root/data/"

python build_data.py \
        --input-dir $data_path/ \
        --output-dir /root/data/databanks/ \
        --vocab-file /root/data/vocab.json