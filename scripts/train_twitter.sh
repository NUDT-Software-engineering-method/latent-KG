#!/bin/bash
home_dir="/home/ubuntu/latent-KG"
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
# no sperate train
for seed in ${seeds[*]}
do
    # -topic_attn_in
    python train_latent_seq2seq.py -data_tag Twitter_s100_t10 -copy_attention -topic_dec -topic_attn  -topic_copy -epochs 110 -early_stop_tolerance 3 -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -batch_workers 0 -seed $seed
done

#topic_nums=(
#20
#50
#100
#150
#200
#500
#1000
#)
#for topic in ${topic_nums[*]}
#do
#    python train_mySeq2Seq.py -data_tag Twitter_s100_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train -topic_dec -topic_attn -topic_attn_in -topic_copy -epochs 110 -early_stop_tolerance 3 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -ntm_warm_up_epochs 50 -use_refs -batch_workers 0 -topic_num $topic
#done

#seeds=(
#10
#100
#5000
#5643
#8000
#9572
#)
#
#for seed in ${seeds[*]}
#do
#    # -topic_dec -topic_attn
#    python train_mySeq2Seq.py -data_tag StackExchange_s150_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train  -topic_attn_in -topic_copy -epochs 110 -early_stop_tolerance 3 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -ntm_warm_up_epochs 20 -use_refs -batch_workers 0 -seed $seed
#done