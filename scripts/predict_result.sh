#!/usr/bin/env bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

dataset="Twitter"
pred_path=(
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed10.emb150.vs30000.dec300.20220216-014630__e53.val_loss=1.515.model-0h-10m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed100.emb150.vs30000.dec300.20220216-015831__e53.val_loss=1.527.model-0h-10m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed5000.emb150.vs30000.dec300.20220216-021032__e53.val_loss=1.519.model-0h-09m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed5643.emb150.vs30000.dec300.20220216-022201__e53.val_loss=1.538.model-0h-10m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed8000.emb150.vs30000.dec300.20220216-023401__e53.val_loss=1.536.model-0h-10m/predictions.txt
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed9572.emb150.vs30000.dec300.20220216-024612__e53.val_loss=1.516.model-0h-09m/predictions.txt
)

for model in ${pred_path[*]}
do
    python pred_evaluate.py -pred $model -src data/${dataset}/test_src.txt -trg data/${dataset}/test_trg.txt
    echo $model
done