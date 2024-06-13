#!/bin/bash

dir="logs_head"
mkdir -p $dir

datasets=("NYC" "SIP" "CHI")
num_nodes_arr=(206 108 220)  # num_nodes for each dataset
tod_size_arr=(48 288 48)     # tod_size for each dataset
heads=(1 2 4 8)
gpus=(0 1 2 3)

for j in "${!datasets[@]}"; do
    dataset=${datasets[$j]}
    num_nodes=${num_nodes_arr[$j]}
    tod_size=${tod_size_arr[$j]}
    
    for i in "${!heads[@]}"; do
        num_heads=${heads[$i]}
        gpu=${gpus[$i]}
        
        python -u main.py \
            --dataset $dataset \
            --gpu $gpu \
            --num_nodes $num_nodes \
            --input_len 12 \
            --output_len 12 \
            --tod_size $tod_size \
            --train_size 0.7 \
            --val_size 0.1 \
            --lr 0.001 \
            --weight_decay 0.0003 \
            --steps 50 70 \
            --gamma 0.1 \
            --batch_size 16 \
            --max_epochs 100 \
            --patience 30 \
            --threshold 0.000001 \
            --obser_dim 3 \
            --output_dim 1 \
            --obser_embed_dim 24 \
            --tod_embed_dim 24 \
            --dow_embed_dim 24 \
            --timestamp_embed_dim 12 \
            --spatial_embed_dim 12 \
            --temporal_embed_dim 60 \
            --prompt_dim 72 \
            --self_atten_dim 168 \
            --cross_atten_dim 24 \
            --feed_forward_dim 256 \
            --num_heads $num_heads \
            --dropout 0.1 \
            >$dir/log_${dataset}_${num_heads}.log &

        sleep 3
    done
    wait
done
wait
