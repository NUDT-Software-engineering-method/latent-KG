home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0
model_path=(
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed10.emb150.vs50000.dec300.20220321-071147/e24.val_loss=2.104.model-1h-24m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed100.emb150.vs50000.dec300.20220321-092804/e24.val_loss=2.125.model-1h-24m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed5000.emb150.vs50000.dec300.20220321-114407/e24.val_loss=2.124.model-1h-24m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed5643.emb150.vs50000.dec300.20220321-135914/e24.val_loss=2.126.model-1h-24m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed8000.emb150.vs50000.dec300.20220321-161418/e24.val_loss=2.109.model-1h-23m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.ntm_warm_up_20.copy.use_refs.seed9572.emb150.vs50000.dec300.20220321-182818/e24.val_loss=2.119.model-1h-24m
)
for model in ${model_path[*]}
do
    python predict_by_newSeq2Seq.py -model $model -batch_size 64 -batch_workers 0
    echo $model
done
# 2116857
# 024715

