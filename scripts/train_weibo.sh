#!/bin/bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

seeds=(
10
100
5000
5643
8000
9572
)

for seed in ${seeds[*]}
do
    python train_mySeq2Seq.py -data_tag Weibo_s100_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train -topic_dec -topic_attn -topic_attn_in -topic_copy -epochs 200 -early_stop_tolerance 3 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -use_refs -ntm_warm_up_epochs 10 -seed $seed -batch_workers 0
    echo $seed
done




