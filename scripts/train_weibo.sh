#!/bin/bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

seeds=(
10
#100
#5000
#5643
#8000
#9572
)

for seed in ${seeds[*]}
do
    python train_latent_seq2seq.py -data_tag Weibo_s100_t10 -copy_attention -topic_dec -topic_attn  -topic_copy -epochs 110 -early_stop_tolerance 3 -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -batch_workers 0 -seed $seed
    echo $seed
done




