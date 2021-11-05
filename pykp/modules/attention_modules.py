import torch
import torch.nn as nn
import torch.nn.functional as F
from pykp.masked_softmax import MaskedSoftmax

class Attention(nn.Module):
    def __init__(self, decoder_size, memory_bank_size, coverage_attn, attn_mode):
        super(Attention, self).__init__()
        # attention
        if attn_mode == "concat":
            self.v = nn.Linear(decoder_size, 1, bias=False)
            self.decode_project = nn.Linear(decoder_size, decoder_size)
        self.memory_project = nn.Linear(memory_bank_size, decoder_size, bias=False)
        self.coverage_attn = coverage_attn
        if coverage_attn:
            self.coverage_project = nn.Linear(1, decoder_size, bias=False)
        self.softmax = MaskedSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.attn_mode = attn_mode

    def score(self, memory_bank, decoder_state, coverage=None):
        """
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param decoder_state: [batch_size, decoder_size]
        :param coverage: [batch_size, max_input_seq_len]
        :return: score: [batch_size, max_input_seq_len]
        """
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        decoder_size = decoder_state.size(1)

        if self.attn_mode == "general":
            # project memory_bank
            memory_bank_ = memory_bank.view(-1,
                                            memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
            """
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                memory_bank_ += self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                memory_bank_ = self.tanh(memory_bank_)

            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            """

            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                encoder_feature += self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                encoder_feature = self.tanh(encoder_feature)

            # expand decoder state
            decoder_state_expanded = decoder_state.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                       decoder_size).contiguous()
            decoder_state_expanded = decoder_state_expanded.view(-1,
                                                                 decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # Perform bi-linear operation
            scores = torch.bmm(decoder_state_expanded.unsqueeze(1),
                               encoder_feature.unsqueeze(2))  # [batch_size*max_input_seq_len, 1, 1]

        else:  # Bahdanau style attention
            # project memory_bank
            memory_bank_ = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            # project decoder state
            dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
            dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                   decoder_size).contiguous()
            dec_feature_expanded = dec_feature_expanded.view(-1,
                                                             decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # sum up attention features
            att_features = encoder_feature + dec_feature_expanded  # [batch_size*max_input_seq_len, decoder_size]

            # Apply coverage
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                coverage_feature = self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                # print(coverage.size())
                # print(coverage_feature.size())
                # print(att_features.size())
                att_features = att_features + coverage_feature

            # compute attention score and normalize them
            e = self.tanh(att_features)  # [batch_size*max_input_seq_len, decoder_size]
            scores = self.v(e)  # [batch_size*max_input_seq_len, 1]

        scores = scores.view(-1, max_input_seq_len)  # [batch_size, max_input_seq_len]
        return scores

    def forward(self, decoder_state, memory_bank, src_mask=None, coverage=None):
        """
        :param decoder_state: [batch_size, decoder_size]
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param src_mask: [batch_size, max_input_seq_len]
        :param coverage: [batch_size, max_input_seq_len]
        :return: context: [batch_size, self.num_directions * self.encoder_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        """
        # init dimension info
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        # decoder_size = decoder_state.size(1)

        if src_mask is None:  # if it does not supply a source mask, create a dummy mask with all ones
            src_mask = memory_bank.new_ones(batch_size, max_input_seq_len)

        """
        # project memory_bank
        memory_bank = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
        encoder_feature = self.memory_project(memory_bank)  # [batch_size*max_input_seq_len, decoder size]

        # project decoder state
        dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
        dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_input_seq_len, decoder_size).contiguous()
        dec_feature_expanded = dec_feature_expanded.view(-1, decoder_size)  # [batch_size*max_input_seq_len, decoder_size]

        # sum up attention features
        att_features = encoder_feature + dec_feature_expanded  # [batch_size*max_input_seq_len, decoder_size]

        # Apply coverage
        if self.coverage_attn:
            coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
            coverage_feature = self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
            #print(coverage.size())
            #print(coverage_feature.size())
            #print(att_features.size())
            att_features = att_features + coverage_feature

        # compute attention score and normalize them
        e = self.tanh(att_features)  # [batch_size*max_input_seq_len, decoder_size]
        scores = self.v(e)  # [batch_size*max_input_seq_len, 1]
        scores = scores.view(-1, max_input_seq_len)  # [batch_size, max_input_seq_len]
        """

        scores = self.score(memory_bank, decoder_state, coverage)
        attn_dist = self.softmax(scores, mask=src_mask)

        # Compute weighted sum of memory bank features
        attn_dist = attn_dist.unsqueeze(1)  # [batch_size, 1, max_input_seq_len]
        memory_bank = memory_bank.view(-1, max_input_seq_len,
                                       memory_bank_size)  # batch_size, max_input_seq_len, memory_bank_size]
        context = torch.bmm(attn_dist, memory_bank)  # [batch_size, 1, memory_bank_size]
        context = context.squeeze(1)  # [batch_size, memory_bank_size]
        attn_dist = attn_dist.squeeze(1)  # [batch_size, max_input_seq_len]

        # Update coverage
        if self.coverage_attn:
            coverage = coverage.view(-1, max_input_seq_len)
            coverage = coverage + attn_dist
            assert coverage.size() == torch.Size([batch_size, max_input_seq_len])

        assert attn_dist.size() == torch.Size([batch_size, max_input_seq_len])
        assert context.size() == torch.Size([batch_size, memory_bank_size])

        return context, attn_dist, coverage

class TopicAttention(nn.Module):
    def __init__(self, decoder_size, memory_bank_size, coverage_attn, attn_mode, topic_num):
        super(TopicAttention, self).__init__()
        # attention
        if attn_mode == "concat":
            self.v = nn.Linear(decoder_size, 1, bias=False)
            self.decode_project = nn.Linear(decoder_size, decoder_size)
        self.topic_num = topic_num
        self.memory_project = nn.Linear(memory_bank_size, decoder_size, bias=False)
        self.topic_project = nn.Linear(topic_num, decoder_size, bias=False)
        self.coverage_attn = coverage_attn
        if coverage_attn:
            self.coverage_project = nn.Linear(1, decoder_size, bias=False)
        self.softmax = MaskedSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.attn_mode = attn_mode

    def score(self, memory_bank, decoder_state, topic_represent, coverage=None):
        """
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param decoder_state: [batch_size, decoder_size]
        :param topic_represent: [batch_size, topic_num]
        :param coverage: [batch_size, max_input_seq_len]
        :return: score: [batch_size, max_input_seq_len]
        """
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        decoder_size = decoder_state.size(1)

        if self.attn_mode == "general":
            # project memory_bank
            memory_bank_ = memory_bank.view(-1,
                                            memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]

            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                encoder_feature += self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                encoder_feature = self.tanh(encoder_feature)

            # expand decoder state
            decoder_state_expanded = decoder_state.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                       decoder_size).contiguous()
            decoder_state_expanded = decoder_state_expanded.view(-1,
                                                                 decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # Perform bi-linear operation
            scores = torch.bmm(decoder_state_expanded.unsqueeze(1),
                               encoder_feature.unsqueeze(2))  # [batch_size*max_input_seq_len, 1, 1]

        else:  # Bahdanau style attention
            # project memory_bank
            memory_bank_ = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            # project decoder state
            topic_feature = self.topic_project(topic_represent)  # [batch_size, decoder_size]
            topic_feature_expanded = topic_feature.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                       decoder_size).contiguous()
            topic_feature_expanded = topic_feature_expanded.view(-1,
                                                                 decoder_size)  # [batch_size*max_input_seq_len, decoder_size]

            dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
            dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                   decoder_size).contiguous()
            dec_feature_expanded = dec_feature_expanded.view(-1,
                                                             decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # sum up attention features
            att_features = encoder_feature + dec_feature_expanded + topic_feature_expanded  # [batch_size*max_input_seq_len, decoder_size]

            # Apply coverage
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                coverage_feature = self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                # print(coverage.size())
                # print(coverage_feature.size())
                # print(att_features.size())
                att_features = att_features + coverage_feature

            # compute attention score and normalize them
            e = self.tanh(att_features)  # [batch_size*max_input_seq_len, decoder_size]
            scores = self.v(e)  # [batch_size*max_input_seq_len, 1]

        scores = scores.view(-1, max_input_seq_len)  # [batch_size, max_input_seq_len]
        return scores

    def forward(self, decoder_state, memory_bank, topic_represent, src_mask=None, coverage=None):
        """
        :param decoder_state: [batch_size, decoder_size]
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param src_mask: [batch_size, max_input_seq_len]
        :param coverage: [batch_size, max_input_seq_len]
        :return: context: [batch_size, self.num_directions * self.encoder_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        """
        # init dimension info
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())

        if src_mask is None:  # if it does not supply a source mask, create a dummy mask with all ones
            src_mask = memory_bank.new_ones(batch_size, max_input_seq_len)

        scores = self.score(memory_bank, decoder_state, topic_represent, coverage)
        attn_dist = self.softmax(scores, mask=src_mask)

        # Compute weighted sum of memory bank features
        attn_dist = attn_dist.unsqueeze(1)  # [batch_size, 1, max_input_seq_len]
        memory_bank = memory_bank.view(-1, max_input_seq_len,
                                       memory_bank_size)  # batch_size, max_input_seq_len, memory_bank_size]
        context = torch.bmm(attn_dist, memory_bank)  # [batch_size, 1, memory_bank_size]
        context = context.squeeze(1)  # [batch_size, memory_bank_size]
        attn_dist = attn_dist.squeeze(1)  # [batch_size, max_input_seq_len]

        # Update coverage
        if self.coverage_attn:
            coverage = coverage.view(-1, max_input_seq_len)
            coverage = coverage + attn_dist
            assert coverage.size() == torch.Size([batch_size, max_input_seq_len])

        assert attn_dist.size() == torch.Size([batch_size, max_input_seq_len])
        assert context.size() == torch.Size([batch_size, memory_bank_size])

        return context, attn_dist, coverage

class ContextTopicAttention(nn.Module):

    def __init__(self, encoder_hidden_size, topic_num, topic_emb_dim, threshold=0.15):
        super(ContextTopicAttention, self).__init__()
        assert encoder_hidden_size == topic_emb_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.topic_num = topic_num
        self.topic_emb_dim = topic_emb_dim
        self.W = nn.Parameter(torch.Tensor(encoder_hidden_size, topic_emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.topic_threshold = threshold

    def forward(self, encoder_memory, attention_dist, topic_emb, topic_dist):
        """
            encoder_memory: [batch_size,seq_len,hidden_dim]
            attention_dist: [batch_size, seq_len]
            topic_emb:      [topic_num, embedding_dim]
            topic_dist:     [batch_size,topic_num]
        """
        batch_size = encoder_memory.shape[0]
        topic_dist = F.softmax(topic_dist, dim=1)
        max_topic_index = torch.argmax(topic_dist, dim=1)  # [batch_size, 1]
        topic_dist = topic_dist - self.topic_threshold

        topic_seq_w = torch.matmul(self.W, topic_emb.T)  # [hidden_size, topic_num]
        seq_topic_w_origin = torch.matmul(encoder_memory, topic_seq_w)  # [batch_size, seq, topic_num]

        topic_seq_w = F.softmax(seq_topic_w_origin, dim=1).permute(0, 2, 1)  # [batch_size, topic_num, seq]
        topic_hidden_state = torch.matmul(topic_seq_w, encoder_memory)  # [batch_size, topic_num, hidden_state]

        # 计算加权的context表示
        topic_dist = topic_dist.unsqueeze(dim=1)
        topic_mean_hidden = torch.matmul(topic_dist, topic_hidden_state)  # [batch_size, 1, hidden_state]
        topic_mean_hidden = topic_mean_hidden.squeeze(dim=1)

        # 计算最相关主题的context表示 怎么按照index 取出来
        batch_size_index = torch.tensor([i for i in range(batch_size)]).to(encoder_memory.device)
        topic_max_hidden = topic_hidden_state[batch_size_index, max_topic_index, :]

        # 计算在每个时间步t下面 加权的topic_embedding
        seq_topic_w = F.softmax(seq_topic_w_origin, dim=2)  # [batch_size, seq, topic_num]
        hidden_topic_state = torch.matmul(seq_topic_w, topic_emb)  # [batch_size, seq, topic_embedding_size]

        return topic_mean_hidden, topic_max_hidden, hidden_topic_state

class TopicEmbeddingAttention(nn.Module):
    def __init__(self, encoder_hidden_size, topic_num, topic_emb_dim):
        super(TopicEmbeddingAttention, self).__init__()
        assert encoder_hidden_size == topic_emb_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.topic_num = topic_num
        self.topic_emb_dim = topic_emb_dim
        self.W = nn.Parameter(torch.Tensor(encoder_hidden_size, topic_emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, encoder_memory, topic_emb):
        """
            encoder_memory: [batch_size,seq_len,hidden_dim]
            attention_dist: [batch_size, seq_len]
            topic_emb:      [topic_num, embedding_dim]
            topic_dist:     [batch_size,topic_num]
        """
        batch_size = encoder_memory.shape[0]

        topic_seq_w = torch.matmul(self.W, topic_emb.T)  # [hidden_size, topic_num]
        seq_topic_w = torch.matmul(encoder_memory, topic_seq_w)  # [batch_size, seq, topic_num]

        # 计算在每个时间步t下面 加权的topic_embedding
        seq_topic_w = F.softmax(seq_topic_w, dim=2)  # [batch_size, seq, topic_num]
        hidden_topic_state = torch.matmul(seq_topic_w, topic_emb)  # [batch_size, seq, topic_embedding_size]

        return hidden_topic_state


class TopicMemeoryMechanism(nn.Module):
    def __init__(self, topic_num, bow_size, embed_size):
        super(TopicMemeoryMechanism, self).__init__()
        self.topic_num = topic_num
        self.bow_size = bow_size
        self.embed_size = embed_size
        self.source_linear = nn.Linear(bow_size, embed_size)
        self.target_linear = nn.Linear(bow_size, embed_size)
        self.embed_project = nn.Linear(embed_size, embed_size)
        self.source_project = nn.Linear(embed_size, embed_size)
        self.weight_p = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, y_embed, topic_word_dist, topic_represent, gamma = 0.8):
        batch_size = y_embed.shape[0]

        y_features = self.embed_project(y_embed)
        y_features = y_features.unsqueeze(1).expand(batch_size, self.topic_num, self.embed_size).contiguous()
        # [batch_size*topic_num, embed_size]
        y_features = y_features.view(-1, self.embed_size)

        # shape: [k, emb_size]
        source_weight = self.relu(self.source_linear(topic_word_dist))
        source_features = self.source_project(source_weight)
        source_features = source_features.unsqueeze(0).expand(batch_size, self.topic_num, self.embed_size).contiguous()
        # [batch_size*topic_num, embed_size]
        source_features = source_features.view(-1, self.embed_size)
        # [batch_size*topic_num, 1]
        p_k_weights = self.sigmoid(self.weight_p(source_features+y_features))
        p_k_weights = p_k_weights.view(batch_size, self.topic_num)

        p_batch = torch.add(gamma*p_k_weights, topic_represent)

        target_weight = self.relu(self.target_linear(topic_word_dist))
        out_embedding = torch.matmul(p_batch, target_weight)
        return p_batch