#!/bin/bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

model_paths=(
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed10.emb150.vs50000.dec300.20211124-222136/e104.val_loss=2.234.model-18h-10m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed100.emb150.vs50000.dec300.20211125-210417/e104.val_loss=2.233.model-17h-29m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed5643.emb150.vs50000.dec300.20211126-190454/e104.val_loss=2.234.model-17h-07m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed8000.emb150.vs50000.dec300.20211127-164247/e104.val_loss=2.244.model-18h-23m
)
for model in ${model_paths[*]}
do
    python predict_by_newSeq2Seq.py -model $model -batch_size 32
done