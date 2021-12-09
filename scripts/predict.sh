home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

model_path=(
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.fusion_emb.use_refs.seed9572.emb150.vs30000.dec300.20211209-105312/e103.val_loss=1.530.model-2h-44m
)
for model in ${model_path[*]}
do
    CUDA_VISIBLE_DEVICES=0 python predict_by_newSeq2Seq.py -model $model -batch_size 64 -batch_workers 0
    echo $model
done