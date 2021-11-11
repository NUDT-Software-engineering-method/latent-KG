import torch
import torch.nn as nn
import torch.nn.functional as F
from pykp.masked_softmax import MaskedSoftmax
from pykp.modules.attention_modules import TopicAttention, Attention, ContextTopicAttention


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, memory_bank_size, coverage_attn, copy_attn,
                 review_attn, pad_idx, attn_mode, dropout=0.0, use_topic_represent=False, topic_attn=False,
                 topic_attn_in=False, topic_copy=False, topic_dec=False, topic_num=50):
        super(RNNDecoder, self).__init__()
        self.use_topic_represent = use_topic_represent
        self.topic_attn = topic_attn
        self.topic_attn_in = topic_attn_in
        self.topic_copy = topic_copy
        self.topic_dec = topic_dec
        self.topic_num = topic_num

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.memory_bank_size = memory_bank_size
        self.dropout = nn.Dropout(dropout)
        self.coverage_attn = coverage_attn
        self.copy_attn = copy_attn
        self.review_attn = review_attn
        self.pad_token = pad_idx
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )

        self.input_size = embed_size

        if use_topic_represent:
            if topic_dec:
                self.input_size = embed_size + topic_num

        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=False, batch_first=False, dropout=dropout)
        if topic_attn_in:
            self.attention_layer = TopicAttention(
                decoder_size=hidden_size,
                memory_bank_size=memory_bank_size,
                coverage_attn=coverage_attn,
                attn_mode=attn_mode,
                topic_num=topic_num
            )
        else:
            self.attention_layer = Attention(
                decoder_size=hidden_size,
                memory_bank_size=memory_bank_size,
                coverage_attn=coverage_attn,
                attn_mode=attn_mode
            )
        if copy_attn:
            if topic_copy:
                self.p_gen_linear = nn.Linear(embed_size + hidden_size + memory_bank_size + topic_num, 1)
            else:
                self.p_gen_linear = nn.Linear(embed_size + hidden_size + memory_bank_size, 1)

        self.sigmoid = nn.Sigmoid()

        if topic_attn:
            self.vocab_dist_linear_1 = nn.Linear(hidden_size + memory_bank_size + topic_num, hidden_size)
        else:
            self.vocab_dist_linear_1 = nn.Linear(hidden_size + memory_bank_size, hidden_size)

        self.vocab_dist_linear_2 = nn.Linear(hidden_size, vocab_size)
        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, y, topic_represent, h, memory_bank, src_mask, max_num_oovs, src_oov, coverage):
        """
        :param y: [batch_size] 表示batch_size个样本在t时间步的单词序号
        :param h: [num_layers, batch_size, decoder_size]
        :param memory_bank: [batch_size, max_src_seq_len, memory_bank_size]
        :param src_mask: [batch_size, max_src_seq_len]
        :param max_num_oovs: int
        :param src_oov: [batch_size, max_src_seq_len]
        :param coverage: [batch_size, max_src_seq_len]
        :return:
        """
        batch_size, max_src_seq_len = list(src_oov.size())
        assert y.size() == torch.Size([batch_size])
        if self.use_topic_represent:
            assert topic_represent.size() == torch.Size([batch_size, self.topic_num])
        assert h.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        # init input embedding
        y_emb = self.embedding(y).unsqueeze(0)  # [1, batch_size, embed_size]

        if self.use_topic_represent and self.topic_dec:
            rnn_input = torch.cat([y_emb, topic_represent.unsqueeze(0)], dim=2)
        else:
            rnn_input = y_emb

        _, h_next = self.rnn(rnn_input, h)

        assert h_next.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        last_layer_h_next = h_next[-1, :, :]  # [batch, decoder_size]

        # apply attention, get input-aware context vector, attention distribution and update the coverage vector
        if self.topic_attn_in:
            context, attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, topic_represent,
                                                                src_mask, coverage)
        else:
            context, attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, src_mask, coverage)
        # context: [batch_size, memory_bank_size]
        # attn_dist: [batch_size, max_input_seq_len]
        # coverage: [batch_size, max_input_seq_len]
        assert context.size() == torch.Size([batch_size, self.memory_bank_size])
        assert attn_dist.size() == torch.Size([batch_size, max_src_seq_len])

        if self.coverage_attn:
            assert coverage.size() == torch.Size([batch_size, max_src_seq_len])

        # 将topic分布加到最后的decoder的输出
        if self.topic_attn:
            vocab_dist_input = torch.cat((context, last_layer_h_next, topic_represent), dim=1)
            # [B, memory_bank_size + decoder_size + topic_num]
        else:
            vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)
            # [B, memory_bank_size + decoder_size]
        vocab_logit = self.vocab_dist_linear_1(vocab_dist_input)
        vocab_dist = self.softmax(self.vocab_dist_linear_2(self.dropout(vocab_logit)))

        p_gen = None
        if self.copy_attn:
            if self.topic_copy:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0), topic_represent),
                                        dim=1)  # [B, memory_bank_size + decoder_size + embed_size]
            else:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)),
                                        dim=1)  # [B, memory_bank_size + decoder_size + embed_size]

            p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if max_num_oovs > 0:
                extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

            final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size])

        return final_dist, h_next, context, attn_dist, p_gen, coverage


class RNNDecoderTW(RNNDecoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, memory_bank_size, coverage_attn, copy_attn,
                 review_attn, pad_idx, attn_mode, bow_size, dropout=0.0, use_topic_represent=False, topic_attn=False,
                 topic_attn_in=False, topic_copy=False, topic_dec=False, topic_num=50):
        super(RNNDecoderTW, self).__init__(vocab_size, embed_size, hidden_size, num_layers, memory_bank_size,
                                           coverage_attn, copy_attn,
                                           review_attn, pad_idx, attn_mode, dropout=dropout,
                                           use_topic_represent=use_topic_represent,
                                           topic_attn=topic_attn,
                                           topic_attn_in=topic_attn_in, topic_copy=topic_copy, topic_dec=topic_dec,
                                           topic_num=topic_num)

        self.bow_size = bow_size
        # self.tm_head = nn.Linear(bow_size, vocab_size, bias=False)
        # self.ada_topic_linear = nn.Linear(hidden_size, topic_num)

        # self.topic_memory = TopicMemeoryMechanism(topic_num=topic_num,bow_size=bow_size,embed_size=embed_size)
        # self.topic_embedding_attention = ContextTopicAttention(encoder_hidden_size=hidden_size, topic_num=topic_num,
        #                                                        topic_emb_dim=hidden_size)
        if copy_attn:
            if topic_copy:
                self.p_gen_linear = nn.Linear(embed_size + hidden_size + memory_bank_size + topic_num, 3)
            else:
                self.p_gen_linear = nn.Linear(embed_size + hidden_size + memory_bank_size, 1)
        self.p_soft_cgate_linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, y, topic_represent, h, memory_bank, hidden_topic_bank, src_mask, max_num_oovs, src_oov, coverage):
        """
        topic_embedding:[topic_num, topic_num_emb]
        """
        batch_size, max_src_seq_len = list(src_oov.size())
        assert y.size() == torch.Size([batch_size])
        if self.use_topic_represent:
            assert topic_represent.size() == torch.Size([batch_size, self.topic_num])
        assert h.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])
        # topic_represent

        # init input embedding
        y_emb = self.embedding(y)
        # topic_represent_embedding = self.topic_memory(y_emb, topic_word, topic_represent)
        y_emb = y_emb.unsqueeze(0)  # [1, batch_size, embed_size]
        if self.use_topic_represent and self.topic_dec:
            rnn_input = torch.cat([y_emb, topic_represent.unsqueeze(0)], dim=2)
        else:
            rnn_input = y_emb
        _, h_next = self.rnn(rnn_input, h)

        assert h_next.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        last_layer_h_next = h_next[-1, :, :]  # [batch, decoder_size]
        # 使用

        # apply attention, get input-aware context vector, attention distribution and update the coverage vector
        if self.topic_attn_in:
            context, attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, topic_represent,
                                                                src_mask, coverage)
            context_topic, attn_dist_topic, coverage_topic = self.attention_layer(last_layer_h_next, hidden_topic_bank,
                                                                                  topic_represent,
                                                                                  src_mask, coverage)
            # 合并两个context 和context_topic
            p_soft_context = self.p_soft_cgate_linear(torch.cat((context, context_topic), dim=1))
            context = p_soft_context * context + (1 - p_soft_context) * context_topic
        else:
            context, attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, src_mask, coverage)
            attn_dist_topic =None
        # 计算topic_embedding之间的attention context
        # topic_context_mean, topic_context_top, seq_topic_weight = self.topic_embedding_attention(memory_bank, attn_dist,
        #                                                                                          topic_embedding,
        #                                                                                          topic_represent)

        # context: [batch_size, memory_bank_size]
        # attn_dist: [batch_size, max_input_seq_len]
        # coverage: [batch_size, max_input_seq_len]
        assert context.size() == torch.Size([batch_size, self.memory_bank_size])
        # assert topic_context_mean.size() == torch.Size([batch_size, self.memory_bank_size])
        # assert topic_context_top.size() == torch.Size([batch_size, self.memory_bank_size])
        assert attn_dist.size() == torch.Size([batch_size, max_src_seq_len])

        if self.coverage_attn:
            assert coverage.size() == torch.Size([batch_size, max_src_seq_len])

        if self.topic_attn:
            vocab_dist_input = torch.cat((context, last_layer_h_next, topic_represent), dim=1)
            # vocab_dist_input_topic = torch.cat((topic_context_top, last_layer_h_next, topic_represent), dim=1)
            # [B, memory_bank_size + decoder_size + topic_num]
        else:
            vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)
            # vocab_dist_input_topic = torch.cat((topic_context_top, last_layer_h_next), dim=1)
            # [B, memory_bank_size + decoder_size]

        # 生成p_gen_topic
        # p_gen_topic = None
        # if self.copy_attn:
        #     if self.topic_copy:
        #         p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0), topic_represent),
        #                                 dim=1)  # [B, memory_bank_size + decoder_size + embed_size]
        #     else:
        #         p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)),
        #                                 dim=1)  # [B, memory_bank_size + decoder_size + embed_size]
        #     p_gen_topic = self.p_gen_topic_linear(p_gen_input)
        #
        # vocab_logit = self.dropout(self.vocab_dist_linear_1(vocab_dist_input))
        # topic_vocab_logit = self.dropout(torch.matmul(self.ada_topic_linear(vocab_logit), self.tm_head(topic_word)))
        # vocab_dist = p_gen_topic * self.vocab_dist_linear_2(vocab_logit) + (1- p_gen_topic)*topic_vocab_logit
        # vocab_dist = self.softmax(vocab_dist)

        vocab_logit = self.vocab_dist_linear_1(vocab_dist_input)
        # vocab_topic_logit = self.vocab_dist_linear_topic_1(vocab_dist_input_topic)

        vocab_dist = self.vocab_dist_linear_2(self.dropout(vocab_logit))

        vocab_dist = self.softmax(vocab_dist)

        p_gen = None
        attn_dist_final = None
        if self.copy_attn:
            if self.topic_copy:
                # TODO : vocab_logit context
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0), topic_represent),
                                        dim=1)  # [B, memory_bank_size + decoder_size + embed_size + topic_num]
            else:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)),
                                        dim=1)  # [B, memory_bank_size + decoder_size + embed_size]

            p_gen = F.softmax(self.p_gen_linear(p_gen_input), dim=1)  # [batch_size, 3]
            # split tensor
            p0, p1, p2 = torch.chunk(p_gen, 3, dim=1)
            vocab_dist_ = p0 * vocab_dist
            attn_dist_ = p1 * attn_dist
            attn_dist_topic_ = p2 * attn_dist_topic
            attn_dist_final = torch.add(attn_dist_, attn_dist_topic_)
            attn_dist_final = self.attention_layer.softmax(attn_dist_final, src_mask)
            if max_num_oovs > 0:
                extra_zeros_topic = attn_dist_topic_.new_zeros((batch_size, max_num_oovs))
                extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

            final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_final)
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size])

        # return final_dist, h_next, context, attn_dist, p_gen, coverage
        return final_dist, h_next, context, attn_dist_final, p_gen, coverage