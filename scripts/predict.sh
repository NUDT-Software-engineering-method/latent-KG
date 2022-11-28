home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0
model_path=(
model/Twitter_s100_t10.copy.seed10.emb150.vs30000.dec300.20221128-165134/e3.val_loss=1.555.model-0h-02m
)
for model in ${model_path[*]}
do
    python predict_by_newSeq2Seq.py -model $model -batch_size 64 -batch_workers 0
    echo $model
done
# 2116857
# 024715

