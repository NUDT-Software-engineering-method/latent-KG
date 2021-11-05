import torch
import torch.nn as nn
import torch.nn.functional as F
from pykp.masked_softmax import MaskedSoftmax
from pykp.modules.attention_modules import ContextTopicAttention, TopicEmbeddingAttention


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token,dropout=0.0):
        super(RNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, src, src_lens):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :return:
        """
        src_embed = self.embedding(src)  # [batch, src_len, embed_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]),
                                                       1)
            # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :]  # [batch, hidden_size]

        return memory_bank.contiguous(), encoder_last_layer_final_state

class AttentionRNNEncoder(RNNEncoder):
    def __init__(self,vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, topic_num, dropout=0.0):
        super(AttentionRNNEncoder, self).__init__(vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=dropout)
        self.topic_embedding_biattention = TopicEmbeddingAttention(encoder_hidden_size=hidden_size*2, topic_num=topic_num, topic_emb_dim=hidden_size*2)
        self.merge_layer = nn.Linear(2*2 * hidden_size,2* hidden_size)

    def forward(self, src, src_lens, topic_embedding):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :return:
        """
        src_embed = self.embedding(src)  # [batch, src_len, embed_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)


        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]),
                                                       1)
            # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :]  # [batch, hidden_size]
        memory_bank = memory_bank.contiguous()
        hidden_topic_state = self.topic_embedding_biattention(memory_bank, topic_embedding)
        hidden_topic_state = torch.cat((hidden_topic_state, memory_bank), dim=2)
        hidden_topic_state = self.merge_layer(hidden_topic_state)

        return memory_bank, encoder_last_layer_final_state, hidden_topic_state