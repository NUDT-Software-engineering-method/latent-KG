home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0
model_path=(
model/kp20ksmall_s400_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_50.copy.use_refs.seed9572.emb150.vs10000.dec300.20220114-164006/e53.val_loss=3.636.model-0h-14m
)
for model in ${model_path[*]}
do
    python predict_by_newSeq2Seq.py -model $model -batch_size 16
    echo $model
done
# 2116857
024715