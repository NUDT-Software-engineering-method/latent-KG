home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

model_path=(
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed9572.emb150.vs30000.dec300.20211221-183452/e54.val_loss=1.552.model-0h-25m
)
for model in ${model_path[*]}
do
    CUDA_VISIBLE_DEVICES=0 python predict_by_newSeq2Seq.py -model $model -batch_size 64
    echo $model
done