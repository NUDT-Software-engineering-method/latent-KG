# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 下午7:12
# @Author  : WuDiDaBinGe
# @FileName: seq2seq_new.py.py
# @Software: PyCharm
import torch
import torch.nn as nn
from pykp.model import Seq2SeqModel
from pykp.modules.contrastive_loss import InstanceLoss
from pykp.modules.attention_modules import ContextTopicAttention
from pykp.modules.ntm import TopicEmbeddingNTM
from pykp.modules.rnn_decoder import RNNDecoderTW, RefRNNDecoder
from pykp.modules.rnn_encoder import AttentionRNNEncoder, RefRNNEncoder, RNNEncoder
from pykp.modules.topic_selector import DocumentTopicDecoder
from torch.nn import functional as F


class TopicSeq2SeqModel(Seq2SeqModel):
    def __init__(self, opt):
        super(TopicSeq2SeqModel, self).__init__(opt)
        self.topic_type = opt.topic_type
        self.encoder_attention = opt.encoder_attention
        self.use_refs = opt.use_refs
        self.use_pretrained = opt.use_pretrained
        self.encoder_dim = opt.encoder_size * self.num_directions
        self.contra_loss = opt.con_loss
        if opt.use_refs:
            self.encoder = RefRNNEncoder.from_opt(opt, self.encoder_embed)
            self.decoder = RefRNNDecoder.from_opt(opt, self.decoder_embed)
        print("Use old ntm model!")
        # self.topic_model = ContextNTM(opt, bert_size=opt.encoder_size * self.num_directions)
        bert_size = self.encoder_dim
        if opt.use_pretrained:
            bert_size = 768
        self.topic_model = TopicEmbeddingNTM(opt, bert_size=bert_size)

        self.topic_num = opt.topic_num
        # 主题注意力机制
        if opt.encoder_attention:
            self.topic_attention = ContextTopicAttention(encoder_hidden_size=self.encoder_dim,
                                                         topic_num=opt.topic_num,
                                                         topic_emb_dim=opt.word_vec_size)

            self.doc_topic_decoder = DocumentTopicDecoder(dim_h=self.encoder_dim,
                                                          num_topics=opt.topic_num)

        # contrastive-loss
        if self.contra_loss:
            self.contra_loss_function = InstanceLoss(opt.device, temperature=0.5)
        # 注意力机制
        self.W = nn.Parameter(torch.Tensor(opt.encoder_size * self.num_directions))
        self.tanh = nn.Tanh()

    def forward(self, src, src_lens, trg=None, src_oov=None, max_num_oov=None, src_mask=None, src_bow=None,
                ref_input=None, begin_iterate_train_ntm=False, num_trgs=None, graph=None):
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

        if self.use_refs:
            if ref_input is None:
                encoder_output, encoder_mask = self.encoder(src, src_lens,
                                                            begin_iterate_train_ntm=begin_iterate_train_ntm)
            else:
                encoder_output, encoder_mask = self.encoder(src, src_lens, ref_docs,
                                                            ref_lens, ref_doc_lens,
                                                            begin_iterate_train_ntm=begin_iterate_train_ntm,
                                                            graph=graph)
            memory_bank, encoder_final_state, encoder_final_state_gat, ref_word_reps, ref_doc_reps = encoder_output
            ref_doc_mask, ref_word_mask = encoder_mask
            if ref_word_mask is not None and ref_word_mask is not None:
                ref_doc_mask = ref_doc_mask.to(src.device)
                ref_word_mask = ref_word_mask.to(src.device)
        else:
            memory_bank, encoder_final_state = self.encoder(src, src_lens)
            encoder_final_state_gat = encoder_final_state
            hidden_topic_state_bank = None
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        topic_context = encoder_final_state

        # Topic Model forward

        topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance = self.topic_model(
            src_bow,
            topic_context)

        if self.topic_type == 'z':
            topic_latent = topic_represent
        else:
            topic_latent = topic_represent_g

        # print(torch.argmax(topic_latent, dim=1)[:15])
        # 只训练主题模型 无需进行解码
        if not begin_iterate_train_ntm:
            # use bi-attention module
            if self.encoder_attention and self.use_contextNTM:
                topic_mean_hidden, topic_max_hidden, hidden_topic_state = self.topic_attention(memory_bank,
                                                                                               self.topic_model.get_topic_embedding(),
                                                                                               topic_latent, src_mask)
                input_doc = topic_mean_hidden
                hidden_doc = torch.zeros((batch_size, self.encoder_dim)).to(src.device)
            h_t_init = self.init_decoder_state(encoder_final_state_gat)  # [dec_layers, batch_size, decoder_size]
            max_target_length = trg.size(1)

            decoder_dist_all = []
            attention_dist_all = []
            decoder_memory_bank = []
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

                if self.encoder_attention:
                    # doc_hidden, topic_dist = self.doc_topic_decoder(input_doc, hidden_doc)
                    # 计算加权的context表示
                    # topic_dist = topic_dist.unsqueeze(dim=1)
                    # topic_mean_hidden = torch.matmul(topic_dist, hidden_topic_state)  # [batch_size, 1, hidden_state]
                    # topic_mean_hidden = topic_mean_hidden.squeeze(dim=1)
                    h_0_sent = topic_mean_hidden
                else:
                    topic_mean_hidden = None
                    doc_hidden = None
                    h_0_sent = None
                if self.use_refs and ref_input is not None:
                    decoder_dist, h_t_next, context, attn_dist, p_gen, coverage = self.decoder(y_t, topic_latent, h_t,
                                                                                               memory_bank, src_mask,
                                                                                               max_num_oov,
                                                                                               src_oov, coverage,
                                                                                               ref_word_reps,
                                                                                               ref_doc_reps,
                                                                                               ref_word_mask,
                                                                                               ref_doc_mask, ref_oovs,
                                                                                               topic_embedding=self.topic_model.get_topic_embedding(),
                                                                                               topic_post_hidden=h_0_sent)

                else:
                    decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                        self.decoder(y_t, topic_latent, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage,
                                     topic_embedding=self.topic_model.get_topic_embedding())

                decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
                attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
                if self.contra_loss:
                    decoder_memory_bank.append(h_t_next.squeeze(0).unsqueeze(1))  # h_t_next: [ batch, 1, decoder_size]

                if self.coverage_attn:
                    coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
                y_t_next = trg[:, t]  # [batch]
                # input_doc = topic_mean_hidden
                # hidden_doc = doc_hidden

            decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
            attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]
            if self.contra_loss:
                decoder_memory_bank = torch.cat(decoder_memory_bank, dim=1)  # [batch_size, trg_len, decoder_size]
            if self.coverage_attn:
                coverage_all = torch.cat(coverage_all, dim=1)  # [batch_size, trg_len, src_len]
                assert coverage_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

            if self.copy_attn:
                assert decoder_dist_all.size() == torch.Size(
                    (batch_size, max_target_length, self.vocab_size + max_num_oov))
            else:
                assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
            assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

            if self.contra_loss:
                masked_memory_bank = memory_bank.masked_fill(src_mask.eq(0).unsqueeze(dim=-1), 0)
                # pool sentence
                encoder_z = F.normalize(torch.mean(masked_memory_bank, dim=1), dim=1)
                decoder_z = F.normalize(torch.mean(decoder_memory_bank, dim=1), dim=1)
                contra_loss = self.contra_loss_function(encoder_z, decoder_z)
            else:
                contra_loss = None
            decoder_output = (
                decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all, contra_loss,
                None, None)

            topic_output = (topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance)
            return decoder_output, topic_output

        decoder_output = (None, None, None, None, None, None, None, None)
        topic_output = (topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance)
        return decoder_output, topic_output


class LatentSeq2SeqModel(Seq2SeqModel):
    def __init__(self, opt):
        super(LatentSeq2SeqModel, self).__init__(opt)

        self.encoder_attention = opt.encoder_attention

        self.use_pretrained = opt.use_pretrained
        self.encoder_dim = opt.encoder_size * self.num_directions
        self.target_encoder = RNNEncoder(
            embed=self.encoder_embed,
            vocab_size=self.vocab_size,
            embed_size=self.emb_dim,
            hidden_size=self.encoder_size,
            num_layers=self.enc_layers,
            bidirectional=self.bidirectional,
            pad_token=self.pad_idx_src,
            dropout=self.dropout
        )
        print("Use old ntm model!")
        # self.topic_model = ContextNTM(opt, bert_size=opt.encoder_size * self.num_directions)
        bert_size = self.encoder_dim
        if opt.use_pretrained:
            bert_size = 768
        self.topic_model = TopicEmbeddingNTM(opt, bert_size=bert_size)

        self.topic_num = opt.topic_num

        self.tanh = nn.Tanh()

    def forward(self, src, src_lens, trg=None, src_oov=None, max_num_oov=None, src_mask=None, src_bow=None,
                ref_input=None, begin_iterate_train_ntm=False, num_trgs=None, graph=None):
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

        memory_bank, encoder_final_state = self.encoder(src, src_lens)
        encoder_final_state_gat = encoder_final_state
        hidden_topic_state_bank = None
        assert memory_bank.size() == torch.Size([batch_size, max_src_len, self.num_directions * self.encoder_size])
        assert encoder_final_state.size() == torch.Size([batch_size, self.num_directions * self.encoder_size])

        topic_context = encoder_final_state

        # Topic Model forward

        topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance = self.topic_model(
            src_bow,
            topic_context)

        if self.topic_type == 'z':
            topic_latent = topic_represent
        else:
            topic_latent = topic_represent_g

        # print(torch.argmax(topic_latent, dim=1)[:15])
        # 只训练主题模型 无需进行解码
        if not begin_iterate_train_ntm:
            # use bi-attention module
            if self.encoder_attention and self.use_contextNTM:
                topic_mean_hidden, topic_max_hidden, hidden_topic_state = self.topic_attention(memory_bank,
                                                                                               self.topic_model.get_topic_embedding(),
                                                                                               topic_latent, src_mask)
                input_doc = topic_mean_hidden
                hidden_doc = torch.zeros((batch_size, self.encoder_dim)).to(src.device)
            h_t_init = self.init_decoder_state(encoder_final_state_gat)  # [dec_layers, batch_size, decoder_size]
            max_target_length = trg.size(1)

            decoder_dist_all = []
            attention_dist_all = []
            decoder_memory_bank = []
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

                if self.encoder_attention:
                    # doc_hidden, topic_dist = self.doc_topic_decoder(input_doc, hidden_doc)
                    # 计算加权的context表示
                    # topic_dist = topic_dist.unsqueeze(dim=1)
                    # topic_mean_hidden = torch.matmul(topic_dist, hidden_topic_state)  # [batch_size, 1, hidden_state]
                    # topic_mean_hidden = topic_mean_hidden.squeeze(dim=1)
                    h_0_sent = topic_mean_hidden
                else:
                    topic_mean_hidden = None
                    doc_hidden = None
                    h_0_sent = None
                if self.use_refs and ref_input is not None:
                    decoder_dist, h_t_next, context, attn_dist, p_gen, coverage = self.decoder(y_t, topic_latent, h_t,
                                                                                               memory_bank, src_mask,
                                                                                               max_num_oov,
                                                                                               src_oov, coverage,
                                                                                               ref_word_reps,
                                                                                               ref_doc_reps,
                                                                                               ref_word_mask,
                                                                                               ref_doc_mask, ref_oovs,
                                                                                               topic_embedding=self.topic_model.get_topic_embedding(),
                                                                                               topic_post_hidden=h_0_sent)

                else:
                    decoder_dist, h_t_next, _, attn_dist, p_gen, coverage = \
                        self.decoder(y_t, topic_latent, h_t, memory_bank, src_mask, max_num_oov, src_oov, coverage,
                                     topic_embedding=self.topic_model.get_topic_embedding())

                decoder_dist_all.append(decoder_dist.unsqueeze(1))  # [batch, 1, vocab_size]
                attention_dist_all.append(attn_dist.unsqueeze(1))  # [batch, 1, src_seq_len]
                if self.contra_loss:
                    decoder_memory_bank.append(h_t_next.squeeze(0).unsqueeze(1))  # h_t_next: [ batch, 1, decoder_size]

                if self.coverage_attn:
                    coverage_all.append(coverage.unsqueeze(1))  # [batch, 1, src_seq_len]
                y_t_next = trg[:, t]  # [batch]
                # input_doc = topic_mean_hidden
                # hidden_doc = doc_hidden

            decoder_dist_all = torch.cat(decoder_dist_all, dim=1)  # [batch_size, trg_len, vocab_size]
            attention_dist_all = torch.cat(attention_dist_all, dim=1)  # [batch_size, trg_len, src_len]
            if self.contra_loss:
                decoder_memory_bank = torch.cat(decoder_memory_bank, dim=1)  # [batch_size, trg_len, decoder_size]
            if self.coverage_attn:
                coverage_all = torch.cat(coverage_all, dim=1)  # [batch_size, trg_len, src_len]
                assert coverage_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

            if self.copy_attn:
                assert decoder_dist_all.size() == torch.Size(
                    (batch_size, max_target_length, self.vocab_size + max_num_oov))
            else:
                assert decoder_dist_all.size() == torch.Size((batch_size, max_target_length, self.vocab_size))
            assert attention_dist_all.size() == torch.Size((batch_size, max_target_length, max_src_len))

            if self.contra_loss:
                masked_memory_bank = memory_bank.masked_fill(src_mask.eq(0).unsqueeze(dim=-1), 0)
                # pool sentence
                encoder_z = F.normalize(torch.mean(masked_memory_bank, dim=1), dim=1)
                decoder_z = F.normalize(torch.mean(decoder_memory_bank, dim=1), dim=1)
                contra_loss = self.contra_loss_function(encoder_z, decoder_z)
            else:
                contra_loss = None
            decoder_output = (
                decoder_dist_all, h_t_next, attention_dist_all, encoder_final_state, coverage_all, contra_loss,
                None, None)

            topic_output = (topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance)
            return decoder_output, topic_output

        decoder_output = (None, None, None, None, None, None, None, None)
        topic_output = (topic_represent, topic_represent_g, recon_x, posterior_mean, posterior_log_variance)
        return decoder_output, topic_output
