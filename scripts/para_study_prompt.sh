#!/bin/bash

dir="logs"
mkdir -p $dir

for dataset in NYC SIP CHI; do
    for prompt_dim in 18 36 72 144; do
        python -u main.py \
            --dataset $dataset \
            --gpu 2 \
            --num_nodes 206 \
            --in_steps 12 \
            --out_steps 12 \
            --steps_per_day 48 \
            --train_size 0.7 \
            --val_size 0.1 \
            --lr 0.001 \
            --weight_decay 0.0003 \
            --milestones 50 120 200 \
            --lr_decay_rate 0.1 \
            --batch_size 16 \
            --max_epochs 500 \
            --patience 30 \
            --threshold 0.000001 \
            --obser_dim 3 \
            --output_dim 1 \
            --obser_embedding_dim 24 \
            --tod_embedding_dim 24 \
            --dow_embedding_dim 24 \
            --timestamp_embedding_dim 12 \
            --spatial_embedding_dim 12 \
            --temporal_embedding_dim 60 \
            --prompt_dim $prompt_dim \
            --self_atten_dim 168 \
            --cross_atten_dim 24 \
            --feed_forward_dim 256 \
            --num_heads 4 \
            --num_layers 1 \
            --dropout 0.1 \
            >$dir/log_${dataset}_${prompt_dim}.log
    done
done
