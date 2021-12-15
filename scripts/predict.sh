home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

model_path=(
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed9527.emb150.vs30000.dec300.20211215-095226/e54.val_loss=1.541.model-3h-24m
)
for model in ${model_path[*]}
do
    CUDA_VISIBLE_DEVICES=0 python predict_by_newSeq2Seq.py -model $model -batch_size 32
    echo $model
done