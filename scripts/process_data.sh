home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

#python preprocess.py -data_dir data/Twitter
#python preprocess.py -data_dir data/Weibo
#python preprocess.py -data_dir data/StackExchange
python preprocess.py -data_dir data/kp20k
