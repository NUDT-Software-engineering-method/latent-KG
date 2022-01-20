#!/usr/bin/env bash
home_dir="/home/ubuntu/TAKG"
export PYTHONPATH=${home_dir}:${PYTHONPATH}

dataset="Twitter"
pred_path=(
pred/test/Full_Dense_RefKP_RefDoc_RefGraph_CopyRef_rnn_One2one_Copy_RefGraph_CopyRef_Seed10_Dropout0.1_LR0.002_BS128_Embed100_NEnc1_NDec1_Dim300/Twitter/predictions.txt
pred/test/Full_Dense_RefKP_RefDoc_RefGraph_CopyRef_rnn_One2one_Copy_RefGraph_CopyRef_Seed100_Dropout0.1_LR0.002_BS128_Embed100_NEnc1_NDec1_Dim300/Twitter/predictions.txt
pred/test/Full_Dense_RefKP_RefDoc_RefGraph_CopyRef_rnn_One2one_Copy_RefGraph_CopyRef_Seed5643_Dropout0.1_LR0.002_BS128_Embed100_NEnc1_NDec1_Dim300/Twitter/predictions.txt
)

for model in ${pred_path[*]}
do
    python pred_evaluate.py -pred $model -src data/${dataset}/test_src.txt -trg data/${dataset}/test_trg.txt
    echo $model
done