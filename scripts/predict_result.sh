#!/usr/bin/env bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

dataset="Twitter"
pred_path=(
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed10.emb150.vs30000.dec300.20211126-074109__e103.val_loss=1.516.model-2h-50m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed100.emb150.vs30000.dec300.20211126-110021__e103.val_loss=1.502.model-2h-49m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed5643.emb150.vs30000.dec300.20211126-141834__e103.val_loss=1.507.model-2h-50m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed8000.emb150.vs30000.dec300.20211126-173747__e104.val_loss=1.510.model-2h-56m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed9572.emb150.vs30000.dec300.20211126-210352__e103.val_loss=1.505.model-2h-49m/predictions.txt
)

for model in ${pred_path[*]}
do
    python pred_evaluate.py -pred $model -src data/${dataset}/test_src.txt -trg data/${dataset}/test_trg.txt
    echo $model
done