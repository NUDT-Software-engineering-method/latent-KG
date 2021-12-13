home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

model_path=(
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed10.emb150.vs30000.dec300.20211212-204341/e54.val_loss=1.525.model-0h-25m
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed100.emb150.vs30000.dec300.20211212-213137/e54.val_loss=1.521.model-0h-25m
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed5643.emb150.vs30000.dec300.20211212-221955/e54.val_loss=1.533.model-0h-25m
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed8000.emb150.vs30000.dec300.20211212-230730/e54.val_loss=1.521.model-0h-25m
)
for model in ${model_path[*]}
do
    CUDA_VISIBLE_DEVICES=0 python predict_by_newSeq2Seq.py -model $model -batch_size 64
    echo $model
done