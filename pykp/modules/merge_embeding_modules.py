# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 下午10:05
# @Author  : WuDiDaBinGe
# @FileName: merge_embeding_modules.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Lda2Vec(nn.Module):
    def __init__(self, word_vec_dim, topic_emb_dim, topic_threshold=0.1, mode="gate_all"):
        super(Lda2Vec, self).__init__()
        self.mode = mode
        if mode == "gate_one":
            self.fusion_layer = nn.Linear(word_vec_dim + topic_emb_dim, 1)
        elif mode == "gate_all":
            assert word_vec_dim == topic_emb_dim
            self.fusion_layer = nn.Linear(word_vec_dim + topic_emb_dim, word_vec_dim)
        self.topic_threshold = topic_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_embedding, topic_embedding, topic_dist):
        batch_size = word_embedding.shape[0]
        topic_dist = F.softmax(topic_dist, dim=1)
        # [batch_size, topic_num]
        topic_dist = topic_dist - self.topic_threshold
        # [batch_size, topic_num] * [topic_num, topic_embedding]
        topic_hidden = torch.matmul(topic_dist, topic_embedding)
        fusion_input = torch.cat((word_embedding, topic_hidden), dim=1)
        merge_embedding = None
        if self.mode == "gate_one":
            # [batch_size, 1]
            p_gen = self.sigmoid(self.fusion_layer(fusion_input))
            word_embedding_ = p_gen * word_embedding
            topic_hidden_ = (1 - p_gen) * topic_hidden
            merge_embedding = word_embedding_ + topic_hidden_
        elif self.mode == "gate_all":
            # [batch_size, vector_dim]
            p_gen_dist = self.sigmoid(self.fusion_layer(fusion_input))
            word_embedding_ = torch.mul(p_gen_dist, word_embedding)
            topic_hidden_ = torch.mul((1 - p_gen_dist), topic_hidden)
            merge_embedding = word_embedding_ + topic_hidden_
        return merge_embedding
