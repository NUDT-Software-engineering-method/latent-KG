# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 下午4:28
# @Author  : WuDiDaBinGe
# @FileName: ntm.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import logging
class NTM(nn.Module):
    def __init__(self, opt, hidden_dim=500, l1_strength=0.001):
        super(NTM, self).__init__()
        self.input_dim = opt.bow_vocab_size
        self.topic_num = opt.topic_num
        topic_num = opt.topic_num
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)
        self.l1_strength = torch.FloatTensor([l1_strength]).to(opt.device)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1

    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        return d1

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar

    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        logging.info("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()

    def get_topic_words(self):
        return self.fcd1.weight.T


class ContextNTM(NTM):
    """
        add encoder last hidden state as input
    """

    def __init__(self, opt, bert_size, hidden_dim=500, l1_strength=0.001):
        super(ContextNTM, self).__init__(opt, hidden_dim, l1_strength)
        self.fc11 = nn.Linear(bert_size + self.input_dim, hidden_dim)
        self.adapt_layer = nn.Linear(bert_size, self.input_dim)

    def encode(self, x, latent_state):
        x_ = x
        x = torch.cat((x, latent_state), 1)
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x_))
        return self.fc21(e1), self.fc22(e1)

    def forward(self, x, latent_state):
        mu, logvar = self.encode(x.view(-1, self.input_dim), latent_state)
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar


class TopicEmbeddingNTM(ContextNTM):
    """
    Add topic word embedding in ntm
    """

    def __init__(self, opt, bert_size, hidden_dim=500, l1_strength=0.001):
        super(TopicEmbeddingNTM, self).__init__(opt, bert_size, hidden_dim=hidden_dim, l1_strength=l1_strength)
        self.topic_embedding_linear = nn.Linear(opt.word_vec_size, self.topic_num, bias=False)
        self.word_embedding_linear = nn.Linear(opt.word_vec_size, self.input_dim, bias=False)
        self.dropout = nn.Dropout(p=opt.dropout)

    def decode(self, z):
        # 这里加上softmax dropout对结果影响不大
        topic_words = self.get_topic_words()
        res = torch.mm(z, topic_words)
        almost_zeros = torch.full_like(res, 1e-6)
        results_without_zeros = res.add(almost_zeros)
        return F.softmax(results_without_zeros, dim=1)

    def forward(self, x, latent_state):
        mu, logvar = self.encode(x.view(-1, self.input_dim), latent_state)
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar

    def get_topic_words(self):
        # word_embedding_d = self.dropout(self.word_embedding)
        # topic_words = torch.matmul(self.topic_embedding, word_embedding_d.T)
        topic_words = self.topic_embedding_linear(self.word_embedding_linear.weight)
        topic_words = topic_words.transpose(1, 0)
        return topic_words

    def print_topic_words(self, vocab_dic, fn, n_top_words=15):
        topic_words = self.get_topic_words()
        beta_exp = topic_words.data.cpu().numpy()
        logging.info("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()

    def get_topic_embedding(self):
        return self.topic_embedding_linear.weight