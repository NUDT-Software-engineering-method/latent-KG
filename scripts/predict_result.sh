#!/usr/bin/env bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

dataset="StackExchange"
pred_path=(
pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.use_ocntm.use_refs.seed10.emb150.vs50000.dec300.20211224-000316__e54.val_loss=2.179.model-1h-55m/predictions.txt
pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.use_ocntm.use_refs.seed100.emb150.vs50000.dec300.20211224-024530__e54.val_loss=2.187.model-1h-55m/predictions.txt
pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.use_ocntm.use_refs.seed5643.emb150.vs50000.dec300.20211224-052700__e54.val_loss=2.178.model-1h-55m/predictions.txt
pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.use_ocntm.use_refs.seed8000.emb150.vs50000.dec300.20211224-080852__e54.val_loss=2.175.model-1h-55m/predictions.txt
pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.use_ocntm.use_refs.seed9572.emb150.vs50000.dec300.20211223-210111__e54.val_loss=2.199.model-1h-56m/predictions.txt
)

for model in ${pred_path[*]}
do
    python pred_evaluate.py -pred $model -src data/${dataset}/test_src.txt -trg data/${dataset}/test_trg.txt
    echo $model
done