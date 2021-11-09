#!/bin/bash
# python preprocess.py -data_dir data/StackExchange
# train pred-ntm
# python train.py -data_tag StackExchange_s150_t10 -only_train_ntm -ntm_warm_up_epochs 100
# 训练论文中的模型
# CUDA_VISIBLE_DEVICES=1 python train.py -data_tag StackExchange_s150_t10 -copy_attention -use_topic_represent -joint_train -topic_dec -topic_copy -topic_attn_in -batch_size 128 -learning_rate 0.002 -check_pt_ntm_model_path model/StackExchange_s150_t10.topic_num50.ntm_warm_up_100.20211102-131637/e90.val_loss=206.676.sparsity=0.851.ntm_model
#&&
# predict by new_seq2seq
CUDA_VISIBLE_DEVICES=1 python predict_by_newSeq2Seq.py -model model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.topic_words.ntm_warm_up_0.copy.useContextNTM.encoder_attention.seed9527.emb150.vs50000.dec300.20211108-040254/e106.val_loss=2.319.model-1h-34m -batch_size 125

# predict by paper model
#CUDA_VISIBLE_DEVICES=1 python predict.py -batch_size 125 -model model/StackExchange_s150_t10.joint_train.use_topic.topic_num50.topic_copy.topic_attn_in.ntm_warm_up_0.copy.seed9527.emb150.vs50000.dec300.20211102-133902/e4.val_loss=2.297.model-0h-05m -ntm_model model/StackExchange_s150_t10.joint_train.use_topic.topic_num50.topic_copy.topic_attn_in.ntm_warm_up_0.copy.seed9527.emb150.vs50000.dec300.20211102-133902/e4.val_loss=2.297.model_ntm-0h-05m

#predict and evaluate
# python pred_evaluate.py -pred pred/predict__StackExchange_s150_t10.joint_train.use_topic.topic_num50.topic_copy.topic_attn_in.ntm_warm_up_0.copy.seed9527.emb150.vs50000.dec300.20211102-133902__e4.val_loss=2.297.model-0h-05m/predictions.txt -src data/StackExchange/test_src.txt -trg data/StackExchange/test_trg.txt


# train on GPU 1 > StackExchange.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1  python train_mySeq2Seq.py -data_tag StackExchange_s150_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train -topic_dec -topic_attn -topic_attn_in -topic_copy -epochs 220 -early_stop_tolerance 10 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -use_contextNTM -encoder_attention -topic_words

