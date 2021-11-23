# -*- coding: utf-8 -*-
# @Time    : 2021/10/22 下午10:42
# @Author  : WuDiDaBinGe
# @FileName: predict_by_newSeq2Seq.py
# @Software: PyCharm
import torch
import logging
import time
import sys
import argparse

import config
from sequence_generator import SequenceGenerator
from utils.time_log import time_since
from evaluate import evaluate_beam_search
from utils.data_loader import load_data_and_vocab
import pykp.io
from pykp.model import Seq2SeqModel, NTM
from pykp.seq2seq_new import TopicSeq2SeqModel
import os
from predict import process_opt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def predict(test_data_loader, model, ntm_model, opt):
    if opt.delimiter_type == 0:
        delimiter_word = pykp.io.SEP_WORD
    else:
        delimiter_word = pykp.io.EOS_WORD
    generator = SequenceGenerator(model,
                                  ntm_model,
                                  opt.use_topic_represent,
                                  opt.topic_type,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  coverage_attn=opt.coverage_attn,
                                  review_attn=opt.review_attn,
                                  length_penalty_factor=opt.length_penalty_factor,
                                  coverage_penalty_factor=opt.coverage_penalty_factor,
                                  length_penalty=opt.length_penalty,
                                  coverage_penalty=opt.coverage_penalty,
                                  cuda=opt.gpuid > -1,
                                  n_best=opt.n_best,
                                  block_ngram_repeat=opt.block_ngram_repeat,
                                  ignore_when_blocking=opt.ignore_when_blocking,
                                  use_topic_words=opt.topic_words,
                                  use_encoder_attention=opt.encoder_attention
                                  )

    evaluate_beam_search(generator, test_data_loader, opt, delimiter_word)


def main(opt):
    try:
        start_time = time.time()
        load_data_time = time_since(start_time)
        test_data_loader, word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=False)
        opt.bow_vocab_size = len(bow_dictionary)
        model = TopicSeq2SeqModel(opt)
        model.load_state_dict(torch.load(opt.model))
        model.to(opt.device)
        model.eval()
        logging.info('Time for loading the data and model: %.1f' % load_data_time)
        start_time = time.time()

        predict(test_data_loader, model, model.topic_model, opt)

        total_testing_time = time_since(start_time)
        logging.info('Time for a complete testing: %.1f' % total_testing_time)
        print('Time for a complete testing: %.1f' % total_testing_time)
        sys.stdout.flush()
    except Exception as e:
        logging.exception("message")
    return


if __name__ == '__main__':
    # load settings for predicting
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_opts(parser)
    config.my_own_opts(parser)
    config.predict_opts(parser)
    config.vocab_opts(parser)

    opt = parser.parse_args()
    opt = process_opt(opt)

    logging = config.init_logging(log_file=opt.pred_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
