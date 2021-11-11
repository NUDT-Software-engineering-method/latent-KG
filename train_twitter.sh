#!/bin/bash
#joint_train: whether to jointly train the model, True or False
#joint_train_strategy: how to jointly train the model, choices=['p_1_iterate', 'p_0_iterate', 'p_1_joint', 'p_0_joint'], iterate: iteratively train each module, joint: jointly train both modules, the digit (0/1): the epoch number of pretraining the seq2seq
#topic_type: use latent variable z or g as topic, g is the default
#topic_dec: add topic in decoder input
#topic_attn: add topic in context vector 和注意力计算的context拼接
#topic_copy: add topic in copy switch    copy机制是否受topic影响
#topic_attn_in: add topic in computing attn score 计算注意力时是否需要加 topic_attn_in
#add_two_loss: use the sum of two losses as the objective
# 训练论文中的模型
#python train.py -data_tag Twitter_s100_t10 -only_train_ntm -ntm_warm_up_epochs 100
#CUDA_VISIBLE_DEVICES=0 python train.py -data_tag Twitter_s100_t10 -copy_attention -use_topic_represent -joint_train -topic_dec -topic_copy -topic_attn_in -batch_size 128 -learning_rate 0.002 -check_pt_ntm_model_path model/Twitter_s100_t10.topic_num50.ntm_warm_up_100.20211105-132919/e90.val_loss=44.481.sparsity=0.848.ntm_model


# train my own model
#CUDA_VISIBLE_DEVICES=0 nohup python train_mySeq2Seq.py -data_tag Weibo_s100_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train -topic_attn -topic_type g -epochs 220 -early_stop_tolerance 50 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -use_contextNTM -topic_num 300 -topic_words > weibo.log 2>&1 &

#CUDA_VISIBLE_DEVICES=1  python train_mySeq2Seq.py -data_tag Twitter_s100_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train -topic_dec -topic_attn -topic_attn_in -topic_copy -epochs 220 -early_stop_tolerance 10 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -use_contextNTM


# python predict
CUDA_VISIBLE_DEVICES=1 python predict_by_newSeq2Seq.py -model model/Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.useContextNTM.seed9527.emb150.vs30000.dec300.20211110-153012/e104.val_loss=1.510.model-0h-18m -batch_size 256

#CUDA_VISIBLE_DEVICES=0 python predict.py -batch_size 125 -model model/Twitter_s100_t10.joint_train.use_topic.topic_num50.topic_copy.topic_attn_in.ntm_warm_up_0.copy.seed9527.emb150.vs30000.dec300.20211105-134742/e2.val_loss=1.547.model-0h-00m -ntm_model model/Twitter_s100_t10.joint_train.use_topic.topic_num50.topic_copy.topic_attn_in.ntm_warm_up_0.copy.seed9527.emb150.vs30000.dec300.20211105-134742/e2.val_loss=1.547.model_ntm-0h-00m
# predict evaluate
#python pred_evaluate.py -pred pred/predict__Twitter_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.topic_words.ntm_warm_up_0.copy.use_ocntm.encoder_attn.seed9527.emb150.vs30000.dec300.20211109-115623__e103.val_loss=1.520.model-0h-19m/predictions.txt -src data/Twitter/test_src.txt -trg data/Twitter/test_trg.txt