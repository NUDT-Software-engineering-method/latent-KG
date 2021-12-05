home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

model_path=(
model/Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed10.emb150.vs50000.dec300.20211126-074055/e104.val_loss=1.224.model-3h-06m
model/Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed100.emb150.vs50000.dec300.20211126-111801/e104.val_loss=1.251.model-3h-05m
model/Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed5643.emb150.vs50000.dec300.20211126-145411/e104.val_loss=1.237.model-3h-07m
model/Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed8000.emb150.vs50000.dec300.20211126-183151/e104.val_loss=1.225.model-3h-07m
model/Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.use_refs.seed9572.emb150.vs50000.dec300.20211126-221012/e104.val_loss=1.235.model-3h-03m
)
for model in ${model_path[*]}
do
    CUDA_VISIBLE_DEVICES=0 python predict_by_newSeq2Seq.py -model $model -batch_size 64 -batch_workers 0
    echo $model
done