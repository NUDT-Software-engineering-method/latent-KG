# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 下午8:26
# @Author  : WuDiDaBinGe
# @FileName: contrastive_loss.py
# @Software: PyCharm
import torch
import torch.nn as nn


class InstanceLoss(nn.Module):
    def __init__(self,  device, temperature=0.07):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        assert z_j.shape == z_i.shape
        batch_size = list(z_i.size())[0]
        self.mask = self.mask_correlated_samples(batch_size)
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        # sim (N * N)
        sim = torch.matmul(z, z.T) / self.temperature
        # sim_i_j (batch_size)
        sim_i_j = torch.diag(sim, batch_size)
        # sim_j_i (batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        # positive (N)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # (N, N-2)
        negative_samples = sim[self.mask].reshape(N, -1)
        # labels  N 第0维的权重应该最大
        labels = torch.zeros(N).to(positive_samples.device).long()
        # logits (N, N-2+1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
