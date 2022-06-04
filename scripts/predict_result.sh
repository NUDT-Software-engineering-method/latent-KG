#!/usr/bin/env bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

dataset="Twitter"
pred_path=(
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.use_refs.seed5000.emb150.vs30000.dec300.20220220-214728
)

for model in ${pred_path[*]}
do
    python pred_evaluate.py -pred $model -src data/${dataset}/test_src.txt -trg data/${dataset}/test_trg.txt
    echo $model
done

#dataset1="StackExchange"
#pred_path1=(
#pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed10.emb150.vs50000.dec300.20220321-071147__e24.val_loss=2.104.model-1h-24m/predictions.txt
#pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed100.emb150.vs50000.dec300.20220321-092804__e24.val_loss=2.125.model-1h-24m/predictions.txt
#pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed5000.emb150.vs50000.dec300.20220321-114407__e24.val_loss=2.124.model-1h-24m/predictions.txt
#pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed5643.emb150.vs50000.dec300.20220321-135914__e24.val_loss=2.126.model-1h-24m/predictions.txt
#pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed8000.emb150.vs50000.dec300.20220321-161418__e24.val_loss=2.109.model-1h-23m/predictions.txt
#pred/predict__SE_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed9572.emb150.vs50000.dec300.20220321-182818__e24.val_loss=2.119.model-1h-24m
#)
#
#for model in ${pred_path1[*]}
#do
#    python pred_evaluate.py -pred $model -src data/${dataset1}/test_src.txt -trg data/${dataset1}/test_trg.txt
#    echo $model
#done