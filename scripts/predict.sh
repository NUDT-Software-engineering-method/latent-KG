home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0
model_path=(
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed10.emb150.vs30000.dec300.20220216-014630/e53.val_loss=1.515.model-0h-10m
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed100.emb150.vs30000.dec300.20220216-015831/e53.val_loss=1.527.model-0h-10m
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed5000.emb150.vs30000.dec300.20220216-021032/e53.val_loss=1.519.model-0h-09m
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed5643.emb150.vs30000.dec300.20220216-022201/e53.val_loss=1.538.model-0h-10m
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed8000.emb150.vs30000.dec300.20220216-023401/e53.val_loss=1.536.model-0h-10m
model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.seed9572.emb150.vs30000.dec300.20220216-024612/e53.val_loss=1.516.model-0h-09m
)
for model in ${model_path[*]}
do
    python predict_by_newSeq2Seq.py -model $model -batch_size 32 -batch_workers 0
    echo $model
done
# 2116857
# 024715