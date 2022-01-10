home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

model_path=(
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_10.copy.use_refs.seed10.emb150.vs50000.dec300.20220109-171412/e14.val_loss=2.166.model-1h-11m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_10.copy.use_refs.seed100.emb150.vs50000.dec300.20220109-191213/e14.val_loss=2.177.model-1h-24m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_10.copy.use_refs.seed5000.emb150.vs50000.dec300.20220109-213416/e14.val_loss=2.204.model-1h-15m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_10.copy.use_refs.seed5643.emb150.vs50000.dec300.20220109-233832/e14.val_loss=2.176.model-1h-12m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_10.copy.use_refs.seed8000.emb150.vs50000.dec300.20220110-013718/e14.val_loss=2.187.model-1h-12m
model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_10.copy.use_refs.seed9572.emb150.vs50000.dec300.20220110-033605/e14.val_loss=2.184.model-1h-12m
)
for model in ${model_path[*]}
do
    CUDA_VISIBLE_DEVICES=0 python predict_by_newSeq2Seq.py -model $model -batch_size 16
    echo $model
done
# 2116857
024715