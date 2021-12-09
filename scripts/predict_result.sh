#!/usr/bin/env bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

dataset="Twitter"
pred_path=(
pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.use_ocntm.use_refs.seed9572.emb150.vs30000.dec300.20211208-151254__e103.val_loss=1.533.model-2h-25m/predictions.txt
)

for model in ${pred_path[*]}
do
    python pred_evaluate.py -pred $model -src data/${dataset}/test_src.txt -trg data/${dataset}/test_trg.txt
    echo $model
done