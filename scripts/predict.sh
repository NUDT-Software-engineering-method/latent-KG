home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

model_path=(
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.useContextNTM.use_refs.seed10.emb150.vs50000.dec300.20211224-000316/e54.val_loss=2.179.model-1h-55m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.useContextNTM.use_refs.seed100.emb150.vs50000.dec300.20211224-024530/e54.val_loss=2.187.model-1h-55m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.useContextNTM.use_refs.seed5643.emb150.vs50000.dec300.20211224-052700/e54.val_loss=2.178.model-1h-55m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.useContextNTM.use_refs.seed8000.emb150.vs50000.dec300.20211224-080852/e54.val_loss=2.175.model-1h-55m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.useContextNTM.use_refs.seed9572.emb150.vs50000.dec300.20211223-210111/e54.val_loss=2.199.model-1h-56m
)
for model in ${model_path[*]}
do
    CUDA_VISIBLE_DEVICES=0 python predict_by_newSeq2Seq.py -model $model -batch_size 32
    echo $model
done
# 2116857
024715