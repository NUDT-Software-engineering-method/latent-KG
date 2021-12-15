#!/usr/bin/env bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

dataset="Weibo"
pred_path=(
pred/predict__Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed10.emb150.vs50000.dec300.20211214-092311__e14.val_loss=1.364.model-0h-23m/predictions.txt
pred/predict__Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed100.emb150.vs50000.dec300.20211214-101426__e14.val_loss=1.337.model-0h-24m/predictions.txt
pred/predict__Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed5643.emb150.vs50000.dec300.20211214-110615__e15.val_loss=1.388.model-0h-29m/predictions.txt
pred/predict__Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed8000.emb150.vs50000.dec300.20211214-120226__e14.val_loss=1.322.model-0h-23m/predictions.txt
pred/predict__Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed9572.emb150.vs50000.dec300.20211214-125229__e14.val_loss=1.339.model-0h-23m/predictions.txt
)

for model in ${pred_path[*]}
do
    python pred_evaluate.py -pred $model -src data/${dataset}/test_src.txt -trg data/${dataset}/test_trg.txt
    echo $model
done