import torch
import torch.nn as nn
import pykp
from pykp.modules.rnn_encoder import RNNEncoder, AttentionRNNEncoder
from pykp.modules.rnn_decoder import RNNDecoder, RNNDecoderTW


class Seq2SeqModel(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()
        self.use_topic_represent = opt.use_topic_represent
        self.topic_num = opt.topic_num
        self.topic_attn = opt.topic_attn
        self.topic_copy = opt.topic_copy
        self.topic_attn_in = opt.topic_attn_in
        self.topic_dec = opt.topic_dec

        self.vocab_size = opt.vocab_size
        self.emb_dim = opt.word_vec_size
        self.num_directions = 2 if opt.bidirectional else 1
        self.encoder_size = opt.encoder_size
        self.decoder_size = opt.decoder_size
        self.batch_size = opt.batch_size
        self.bidirectional = opt.bidirectional
        self.enc_layers = opt.enc_layers
        self.dec_layers = opt.dec_layers
        self.dropout = opt.dropout

        self.bridge = opt.bridge
        self.one2many_mode = opt.one2many_mode
        self.one2many = opt.one2many

        self.coverage_attn = opt.coverage_attn
        self.copy_attn = opt.copy_attention

        self.pad_idx_src = opt.word2idx[pykp.io.PAD_WORD]
        self.pad_idx_trg = opt.word2idx[pykp.io.PAD_WORD]
        self.bos_idx = opt.word2idx[pykp.io.BOS_WORD]
        self.eos_idx = opt.word2idx[pykp.io.EOS_WORD]
        self.unk_idx = opt.word2idx[pykp.io.UNK_WORD]
        self.sep_idx = opt.word2idx[pykp.io.SEP_WORD]
        self.orthogonal_loss = opt.orthogonal_loss

        self.share_embeddings = opt.share_embeddings
        self.review_attn = opt.review_attn

        self.attn_mode = opt.attn_mode

        self.device = opt.device
        # 定义encoder和decoder共享的embedding
        self.embed = nn.Embedding(
            self.vocab_size,
            self.emb_dim,
            self.pad_idx_src
        )
        self.encoder = RNNEncoder(
            embed=self.embed,
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.encoder_size,
            num_layers=self.enc_layers,
            bidirectional=self.bidirectional,
            pad_token=self.pad_idx_src,
            dropout=self.dropout
        )

        self.decoder = RNNDecoder(
            embed=self.embed,
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.decoder_size,
            num_layers=self.dec_layers,
            memory_bank_size=self.num_directions * self.encoder_size,
            coverage_attn=self.coverage_attn,
            copy_attn=self.copy_attn,
            review_attn=self.review_attn,
            pad_idx=self.pad_idx_trg,
            attn_mode=self.attn_mode,
            dropout=self.dropout,
            use_topic_represent=self.use_topic_represent,  # yue
            topic_attn=self.topic_attn,
            topic_attn_in=self.topic_attn_in,
            topic_copy=self.topic_copy,
            topic_dec=self.topic_dec,
            topic_num=self.topic_num
        )

        if self.bridge == 'dense':
            self.bridge_layer = nn.Linear(self.encoder_size * self.num_directions, self.decoder_size)
        elif opt.bridge == 'dense_nonlinear':
            self.bridge_layer = torch.tanh(nn.Linear(self.encoder_size * self.num_directions, self.decoder_size))
        else:
            self.bridge_layer = None

        if self.bridge == 'copy':
            assert self.encoder_size * self.num_directions == self.decoder_size, \
                'encoder hidden size and decoder hidden size are not match, please use a bridge layer'

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, topic_represent, num_trgs=None, **kwargs):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :param num_trgs: only effective in one2many mode 2, a list of num of targets in each batch, with len=batch_size
        :return:
        """
        batch_size, max_src_len = list(src.size())

        # Encoding
        memory_bank, encoder_final_state = self.encoder(src, src_lens)
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        # Decoding
        h_t_init = self.init_decoder_state(encoder_final_state)  # [dec_layers, batch_size, decoder_size]
        max_target_length = trg.size(1)

        decoder_dist_all = []
        attention_dist_all = []

        if self.coverage_attn:
            coverage = torch.zeros_like(src, dtype=torch.float).requires_grad_()  # [batch, max_src_seq]
            coverage_all = []
        else:
            coverage = None
            coverage_all = None

        # init y_t to be BOS token
        y_t_init = trg.new_ones(batch_size) * self.bos_idx  # [batch_size]

        for t in range(max_target_length):
            if t == 0:
                h_t = h_t_init
                y_t = y_t_init
            else:
                h_t = h_t_next
                y_t = y_t_next

            decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                self.decoder(y_t, topic_represent, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage)
            decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
            attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
            if self.coverage_attn:
                coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
            y_t_next = trg[:, t]  # [batch]

        decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
        attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]
        if self.coverage_attn:
            coverage_all = torch.cat(coverage_all, dim=1)  # [batch_size, trg_len, src_len]
            assert coverage_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        if self.copy_attn:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size + max_num_oov))
        else:
            assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
        assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

        return decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all, None, None, None

    def init_decoder_state(self, encoder_final_state):
        """
        :param encoder_final_state: [batch_size, self.num_directions * self.encoder_size]
        :return: [1, batch_size, decoder_size]
        """
        batch_size = encoder_final_state.size(0)
        if self.bridge == 'none':
            decoder_init_state = None
        elif self.bridge == 'copy':
            decoder_init_state = encoder_final_state
        else:
            decoder_init_state = self.bridge_layer(encoder_final_state)
        decoder_init_state = decoder_init_state.unsqueeze(0).expand((self.dec_layers, batch_size, self.decoder_size))
        # [dec_layers, batch_size, decoder_size]
        return decoder_init_state

    def init_context(self, memory_bank):
        # Init by max pooling, may support other initialization later
        context, _ = memory_bank.max(dim=1)
        return context
