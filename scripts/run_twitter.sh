home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0
data_dir="data/Twitter"

ref_doc_path="${data_dir}/train_src.txt"
ref_kp_path="${data_dir}/train_trg.txt"
hash_path="${data_dir}/train_src-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"


echo "============================= build_tfidf ================================="

cmd="python retrievers/build_tfidf.py \
-ref_doc_path ${ref_doc_path} \
-out_dir ${data_dir}
"

echo $cmd
eval $cmd