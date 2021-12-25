# -*- coding: utf-8 -*-
# @Time    : 2021/12/22 下午8:37
# @Author  : WuDiDaBinGe
# @FileName: topic_selector.py
# @Software: PyCharm
from torch import nn
import torch

class DocumentTopicDecoder(nn.Module):
    def __init__(self, dim_h, num_topics):
        super(DocumentTopicDecoder, self).__init__()
        self.decoder = nn.GRUCell(input_size=dim_h, hidden_size=dim_h)
        self.out_linear = nn.Linear(dim_h, num_topics)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        """
        Args:
            - input (bsz, dim_h)
            - hidden (bsz, dim_h)
            - avail_topic_mask (bsz, num_topics)
        Return:
            - hidden_out (bsz, dim_h) : hidden state of this step
            - topic_dist (bsz, num_topics) : probablity distribution of next sentence on topics
        """
        hidden_out = self.decoder(input, hidden)
        topic_dist = self.out_linear(hidden_out)
        topic_dist = self.softmax(topic_dist)
        # print(torch.argmax(topic_dist, dim=1)[:10])
        return hidden_out, topic_dist
