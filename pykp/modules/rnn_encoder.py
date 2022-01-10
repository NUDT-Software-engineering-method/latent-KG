import torch
import torch.nn as nn
import torch.nn.functional as F

import pykp
from pykp.masked_softmax import MaskedSoftmax
from pykp.modules.attention_modules import ContextTopicAttention, TopicEmbeddingAttention
from pykp.modules.gat import GAT


class RNNEncoder(nn.Module):
    def __init__(self, embed, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
        super(RNNEncoder, self).__init__()
        self.embedding = embed
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)

    @classmethod
    def from_opt(cls, opt, embed):
        return cls(embed=embed,
                   vocab_size=opt.opt.vocab_size,
                   embed_size=opt.word_vec_size,
                   hidden_size=opt.encoder_size,
                   num_layers=opt.enc_layers,
                   bidirectional=opt.bidirectional,
                   pad_token=opt.word2idx[pykp.io.PAD_WORD],
                   dropout=opt.dropout)

    def _forward(self, rnn, src, src_lens):
        """
            :param: rnn: enn encoder
            :param src: [batch, src_seq_len]
            :param src_lens: a list containing the length of src sequences for each batch, with len=batch
            :return:
        """
        src_embed = self.embedding(src)  # [batch, src_len, embed_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True,
                                                             enforce_sorted=False)
        memory_bank, encoder_final_state = rnn(packed_input_src)
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

    def forward(self, src, src_lens, **kwargs):
        memory_bank, encoder_last_final_state = self._forward(self.rnn, src, src_lens)
        return memory_bank, encoder_last_final_state


class AttentionRNNEncoder(RNNEncoder):
    def __init__(self, embed, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, topic_num,
                 topic_embed_dim, dropout=0.0):
        super(AttentionRNNEncoder, self).__init__(embed, vocab_size, embed_size, hidden_size, num_layers, bidirectional,
                                                  pad_token, dropout=dropout)
        self.topic_embedding_biattention = TopicEmbeddingAttention(encoder_hidden_size=hidden_size * 2,
                                                                   topic_num=topic_num, topic_emb_dim=topic_embed_dim)
        self.merge_layer = nn.Linear(2 * hidden_size + topic_embed_dim, 2 * hidden_size)

    @classmethod
    def from_opt(cls, opt, embed):
        return cls(embed=embed,
                   vocab_size=opt.opt.vocab_size,
                   embed_size=opt.word_vec_size,
                   hidden_size=opt.encoder_size,
                   num_layers=opt.enc_layers,
                   bidirectional=opt.bidirectional,
                   pad_token=opt.word2idx[pykp.io.PAD_WORD],
                   dropout=opt.dropout)

    def forward(self, src, src_lens, topic_embedding=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :param topic_embedding: a topic_embedding matrix [topic_num. topic_embedding_dim]
        :return:
        """
        assert topic_embedding is not None
        src_embed = self.embedding(src)  # [batch, src_len, embed_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)

        # 计算与topic embedding的注意力
        memory_bank = memory_bank.contiguous()
        hidden_topic_state = self.topic_embedding_biattention(memory_bank,
                                                              topic_embedding)  # [batch_size, seq, topic_embedding_dim]
        hidden_topic_state = torch.cat((hidden_topic_state, memory_bank),
                                       dim=2)  # [batch_size, seq, 2*hidden + topic_embedding_dim]
        hidden_topic_state = F.tanh(self.merge_layer(hidden_topic_state))  # [batch_size, seq, 2*hidden]
        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]),
                                                       1)
            # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :]  # [batch, hidden_size]

        # 使用

        return memory_bank, encoder_last_layer_final_state, hidden_topic_state


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    length: (batchsize,)
    """
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(*lengths.size(), 1)
            .lt(lengths.unsqueeze(-1)))


class RefRNNEncoder(RNNEncoder):
    def __init__(self, embed, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, opt,
                 dropout=0.0):
        super(RefRNNEncoder, self).__init__(embed, vocab_size, embed_size, hidden_size, num_layers, bidirectional,
                                            pad_token,
                                            dropout=dropout)
        self.rnn_ref = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers + 1,
                              bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.gat = GAT.from_opt(opt, self.embedding)

    @classmethod
    def from_opt(cls, opt, embed):
        return cls(embed=embed,
                   vocab_size=opt.vocab_size,
                   embed_size=opt.word_vec_size,
                   hidden_size=opt.encoder_size,
                   num_layers=opt.enc_layers,
                   bidirectional=opt.bidirectional,
                   pad_token=opt.word2idx[pykp.io.PAD_WORD],
                   opt=opt,
                   dropout=opt.dropout)

    def forward(self, src, src_lens, ref_docs=None, ref_lens=None, ref_doc_lens=None, begin_iterate_train_ntm=False,
                graph=None):
        cur_word_rep, cur_doc_rep = self._forward(self.rnn, src, src_lens)
        cur_doc_rep_gat = None
        if not begin_iterate_train_ntm:
            packed_ref_docs_by_ref_lens = nn.utils.rnn.pack_padded_sequence(ref_docs, ref_lens, batch_first=True,
                                                                            enforce_sorted=False)
            packed_doc_ref_lens_by_ref_lens = nn.utils.rnn.pack_padded_sequence(ref_doc_lens, ref_lens,
                                                                                batch_first=True, enforce_sorted=False)

            packed_ref_word_reps, packed_ref_doc_reps = self._forward(self.rnn_ref, packed_ref_docs_by_ref_lens.data,
                                                                      packed_doc_ref_lens_by_ref_lens.data.cpu())
            ref_word_reps, _ = nn.utils.rnn.pad_packed_sequence(
                nn.utils.rnn.PackedSequence(data=packed_ref_word_reps,
                                            batch_sizes=packed_ref_docs_by_ref_lens.batch_sizes,
                                            sorted_indices=packed_ref_docs_by_ref_lens.sorted_indices,
                                            unsorted_indices=packed_ref_docs_by_ref_lens.unsorted_indices),
                batch_first=True
            )  # [batch, max_doc_num, max_doc_len, hidden_size]
            ref_doc_reps, _ = nn.utils.rnn.pad_packed_sequence(
                nn.utils.rnn.PackedSequence(data=packed_ref_doc_reps,
                                            batch_sizes=packed_ref_docs_by_ref_lens.batch_sizes,
                                            sorted_indices=packed_ref_docs_by_ref_lens.sorted_indices,
                                            unsorted_indices=packed_ref_docs_by_ref_lens.unsorted_indices),
                batch_first=True
            )  # [batch, max_doc_num, hidden_size]

            if graph is not None:
                assert graph is not None
                all_doc_rep = torch.cat([cur_doc_rep.unsqueeze(1), ref_doc_reps], 1)
                all_doc_rep = self.gat(graph, ref_lens + 1, all_doc_rep)
                cur_doc_rep_gat = all_doc_rep[:, 0].contiguous()
                ref_doc_reps = all_doc_rep[:, 1:].contiguous()

            ref_doc_mask = sequence_mask(ref_lens)
            ref_word_mask = sequence_mask(ref_doc_lens)
        else:
            ref_word_reps, ref_doc_reps, ref_doc_mask, ref_word_mask = None, None, None, None
        return (cur_word_rep, cur_doc_rep, cur_doc_rep_gat, ref_word_reps, ref_doc_reps), (ref_doc_mask, ref_word_mask)
