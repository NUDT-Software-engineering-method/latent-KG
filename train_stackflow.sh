#!/bin/bash
# python preprocess.py -data_dir data/StackExchange
# train pred-ntm
# python train.py -data_tag StackExchange_s150_t10 -only_train_ntm -ntm_warm_up_epochs 100
# train joint
# python train.py -data_tag StackExchange_s150_t10 -copy_attention -use_topic_represent -load_pretrain_ntm -joint_train -topic_attn -check_pt_ntm_model_path model/StackExchange_s150_t10.topic_num50.ntm_warm_up_100.20211021-171355/e100.val_loss=206.586.sparsity=0.850.ntm_model
#&&
# predict
#CUDA_VISIBLE_DEVICES=1 python predict_by_newSeq2Seq.py -model model/StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.no_topic_dec.ntm_warm_up_0.copy.seed9527.emb150.vs50000.dec300.20211101-142415/e106.val_loss=2.420.model-1h-43m -batch_size 125 -topic_type g -use_contextNTM -topic_words
#predict and evaluate
# python pred_evaluate.py -pred pred/predict__StackExchange_s150_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.no_topic_dec.ntm_warm_up_0.copy.seed9527.emb150.vs50000.dec300.20211101-142415__e106.val_loss=2.420.model-1h-43m/predictions.txt -src data/StackExchange/test_src.txt -trg data/StackExchange/test_trg.txt
# train on GPU 1
CUDA_VISIBLE_DEVICES=1  python train_mySeq2Seq.py -data_tag StackExchange_s150_t10 -copy_attention -use_topic_represent -add_two_loss -joint_train -topic_attn -topic_type g -epochs 220 -early_stop_tolerance 5 -joint_train_strategy p_1_iterate -learning_rate_decay 0.9 -batch_size 128 -learning_rate 0.002 -use_contextNTM -topic_words
