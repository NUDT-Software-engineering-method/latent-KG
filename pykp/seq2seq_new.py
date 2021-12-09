# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 下午7:12
# @Author  : WuDiDaBinGe
# @FileName: seq2seq_new.py.py
# @Software: PyCharm
import torch
import torch.nn as nn
from pykp.model import Seq2SeqModel
from pykp.context_topic_model.decoding_network import DecoderNetwork
from pykp.modules.ntm import TopicEmbeddingNTM
from pykp.modules.rnn_decoder import RNNDecoderTW, RefRNNDecoder
from pykp.modules.rnn_encoder import AttentionRNNEncoder, RefRNNEncoder


class TopicSeq2SeqModel(Seq2SeqModel):
    def __init__(self, opt):
        super(TopicSeq2SeqModel, self).__init__(opt)
        self.topic_type = opt.topic_type
        self.use_contextNTM = opt.use_contextNTM
        self.encoder_attention = opt.encoder_attention

        if opt.encoder_attention:
            self.encoder = AttentionRNNEncoder.from_opt(opt, self.embed)
        self.topic_words = opt.topic_words
        if opt.topic_words:
            self.decoder = RNNDecoderTW.from_opt(opt, self.embed)
        self.use_refs = opt.use_refs
        self.use_pretrained = opt.use_pretrained

        if opt.use_refs:
            self.encoder = RefRNNEncoder.from_opt(opt, self.embed)
            self.decoder = RefRNNDecoder.from_opt(opt, self.embed)
        if self.use_contextNTM:
            print("Use old ntm model!")
            # self.topic_model = ContextNTM(opt, bert_size=opt.encoder_size * self.num_directions)
            bert_size = opt.encoder_size * self.num_directions
            if opt.use_pretrained:
                bert_size = 768
            self.topic_model = TopicEmbeddingNTM(opt, bert_size=bert_size)

        else:
            print("Use new ntm model!")
            self.topic_model = DecoderNetwork(vocab_size=opt.bow_vocab_size,
                                              bert_size=opt.encoder_size * self.num_directions, infnet="combined",
                                              num_topics=opt.topic_num, model_type="prodLDA", hidden_sizes=(100, 100),
                                              activation="softplus", dropout=opt.dropout, learn_priors=True)

        # 注意力机制
        self.W = nn.Parameter(torch.Tensor(opt.encoder_size * self.num_directions))
        self.tanh = nn.Tanh()

    def forward(self, src, src_lens, trg, src_oov, max_num_oov, src_mask, src_bow, query_emb=None, ref_input=None,
                begin_iterate_train_ntm=False, num_trgs=None, graph=None):
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
        ref_docs, ref_lens, ref_doc_lens, ref_oovs, encoder_final_state_gat = None, None, None, None, None
        if ref_input is not None:
            ref_docs, ref_lens, ref_doc_lens, ref_oovs = ref_input

        # Encoding
        if self.encoder_attention:
            memory_bank, encoder_final_state, hidden_topic_state_bank = self.encoder(src, src_lens,
                                                                                     self.topic_model.get_topic_embedding())
        elif self.use_refs and ref_input is not None:
            encoder_output, encoder_mask = self.encoder(src, src_lens, ref_docs,
                                                        ref_lens, ref_doc_lens,
                                                        begin_iterate_train_ntm=begin_iterate_train_ntm, graph=graph)
            memory_bank, encoder_final_state, encoder_final_state_gat, ref_word_reps, ref_doc_reps = encoder_output
            ref_doc_mask, ref_word_mask = encoder_mask
            if ref_word_mask is not None and ref_word_mask is not None:
                ref_doc_mask = ref_doc_mask.to(src.device)
                ref_word_mask = ref_word_mask.to(src.device)
        else:
            memory_bank, encoder_final_state = self.encoder(src, src_lens)
            hidden_topic_state_bank = None
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        # 判断是否为使用预训练的向量作为主题模型的输入
        if query_emb is None:
            topic_context = encoder_final_state
        else:
            topic_context = query_emb
        # Topic Model forward
        if self.use_contextNTM:
            topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance = self.topic_model(
                src_bow,
                topic_context)
        else:
            topic_represent, topic_represent_g, recon_x, (
                posterior_mean, posterior_variance, posterior_log_variance), (
                prior_mean, prior_variance) = self.topic_model(
                src_bow, topic_context)
        if self.topic_type == 'z':
            topic_latent = topic_represent
        else:
            topic_latent = topic_represent_g

        # 只训练主题模型 无需进行解码
        if not begin_iterate_train_ntm:
            h_t_init = self.init_decoder_state(encoder_final_state_gat)  # [dec_layers, batch_size, decoder_size]
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
                if self.topic_words and self.encoder_attention:
                    # decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                    #     self.decoder(y_t, topic_latent, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage,
                    #                  self.topic_model.get_topic_words())
                    decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                        self.decoder(y_t, topic_latent, h_t, memory_bank, hidden_topic_state_bank, src_mask,
                                     max_num_oov, src_oov, coverage, topic_embedding=self.topic_model.get_topic_embedding())
                elif self.use_refs and ref_input is not None:
                    decoder_dist, h_t_next, context, attn_dist, p_gen, coverage = self.decoder(y_t, topic_latent, h_t,
                                                                                               memory_bank, src_mask,
                                                                                               max_num_oov,
                                                                                               src_oov, coverage,
                                                                                               ref_word_reps,
                                                                                               ref_doc_reps,
                                                                                               ref_word_mask,
                                                                                               ref_doc_mask, ref_oovs, topic_embedding=self.topic_model.get_topic_embedding())

                else:
                    decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                        self.decoder(y_t, topic_latent, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage, topic_embedding=self.topic_model.get_topic_embedding())

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
                assert decoder_dist_all.size() == torch.Size(
                    (batch_size, max_target_length, self.vocab_size + max_num_oov))
            else:
                assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
            assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))
            decoder_output = (
                decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all, None, None, None)
            if self.use_contextNTM:
                topic_output = (topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance)
                return decoder_output, topic_output
            else:
                topic_out = (topic_represent, topic_represent_g, recon_x,
                             (posterior_mean, posterior_variance, posterior_log_variance), (prior_mean, prior_variance))
                return decoder_output, topic_out
        decoder_output = (None, None, None, None, None, None, None, None)
        if self.use_contextNTM:
            topic_output = (topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance)
            return decoder_output, topic_output
        else:
            topic_out = (topic_represent, topic_represent_g, recon_x,
                         (posterior_mean, posterior_variance, posterior_log_variance), (prior_mean, prior_variance))
            return decoder_output, topic_out
