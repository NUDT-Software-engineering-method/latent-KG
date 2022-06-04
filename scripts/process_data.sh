home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=1
# -dense_retrieve
# -hash_path data/Twitter/train_src-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
# python preprocess.py -data_dir data/Twitter -dense_retrieve

python preprocess.py -data_dir data/Weibo -dense_retrieve
#python preprocess.py -data_dir data/StackExchange -dense_retrieve
#python preprocess.py -data_dir data/kp20k -dense_retrieve
