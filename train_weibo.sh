#!/bin/bash

# train joint
#CUDA_VISIBLE_DEVICES=0 nohup python train_mySeq2Seq.py -data_tag Weibo_s100_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train -topic_attn -topic_type g -epochs 220 -early_stop_tolerance 50 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -use_contextNTM -topic_num 300 -topic_words > weibo.log 2>&1 &
CUDA_VISIBLE_DEVICES=0  python train_mySeq2Seq.py -data_tag Weibo_s100_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train -topic_attn -topic_type g -epochs 220 -early_stop_tolerance 5 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002  -topic_words -use_contextNTM

# python predict
#CUDA_VISIBLE_DEVICES=0 python predict_by_newSeq2Seq.py -model model/Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.no_topic_dec.ntm_warm_up_0.copy.seed9527.emb150.vs50000.dec300.20211101-142323/e105.val_loss=1.437.model-0h-31m -batch_size 125 -topic_type g -use_contextNTM -topic_words

# predict evaluate
#python pred_evaluate.py -pred pred/predict__Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.no_topic_dec.ntm_warm_up_0.copy.seed9527.emb150.vs50000.dec300.20211101-142323__e105.val_loss=1.437.model-0h-31m/predictions.txt -src data/Weibo/test_src.txt -trg data/Weibo/test_trg.txt