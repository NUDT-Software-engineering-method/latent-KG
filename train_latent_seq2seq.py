# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 下午9:21
# @Author  : WuDiDaBinGe
# @FileName: train_latent_seq2seq.py
# @Software: PyCharm
import argparse
from torch.optim import Adam

import config
from train_mixture import loss_function, EPS, \
    fix_model_seq2seq_decoder, unfix_model_seq2seq_decoder, l1_penalty, check_sparsity, update_l1
from pykp.seq2seq_new import TopicSeq2SeqModel, LatentSeq2SeqModel
from train import process_opt
from utils.data_loader import load_data_and_vocab
from pykp.context_topic_model.loss import topic_modeling_loss
import torch.nn as nn
from torch.nn import functional as F
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.time_log import time_since, convert_time2str
from evaluate import evaluate_loss
import time
import math
import logging
import torch
import sys
import os
from tensorboardX import SummaryWriter
import torch.multiprocessing


# torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_optimizers(model, opt):
    # optimizer_seq2seq = Adam(
    #     params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    # optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, model.topic_model.parameters()), lr=opt.learning_rate)
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)

    return optimizer_whole


def evaluate_loss(data_loader, topic_seq2seqModel, opt):
    topic_seq2seqModel.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    print("Evaluate loss for %d batches" % len(data_loader))
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            if not opt.one2many:  # load one2one dataset
                src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, src_bow, \
                ref_docs, ref_lens, ref_doc_lens, ref_oovs, graph = batch
            else:  # load one2many dataset
                src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = batch
                num_trgs = [len(trg_str_list) for trg_str_list in
                            trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)

            src_bow = src_bow.to(opt.device)
            src_bow_norm = F.normalize(src_bow)
            start_time = time.time()

            # for one2one setting
            ref_input = (ref_docs, ref_lens, ref_doc_lens, ref_oovs)
            seq2seq_output, topic_model_output = topic_seq2seqModel(src, src_lens, trg, src_oov, max_num_oov, src_mask,
                                                                    src_bow_norm, ref_input=ref_input, graph=graph)
            decoder_dist, h_t, attention_dist, encoder_final_state, coverage = seq2seq_output
            quantized_latent, vq_loss, rec_bow = topic_model_output

            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            if opt.copy_attention:  # Compute the loss using target with oov words
                loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                            coverage_loss=False)
            else:  # Compute the loss using target without oov words
                loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                            coverage_loss=False)

            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)

            if (batch_i + 1) % (len(data_loader) // 5) == 0:
                print("Train: %d/%d batches, current avg loss: %.3f" %
                      ((batch_i + 1), len(data_loader), evaluation_loss_sum / total_trg_tokens))

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def train_one_batch(batch, topic_seq2seq_model, optimizer, opt, batch_i):
    # train for one batch
    #  begin_iterate_train_ntm whether train ntm only
    src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, src_bow, \
    ref_docs, ref_lens, ref_doc_lens, ref_oovs, graph = batch
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)
    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)

    # graph = graph.to(opt.device)
    # model.train()
    optimizer.zero_grad()

    start_time = time.time()
    src_bow = src_bow.to(opt.device)
    src_bow_norm = F.normalize(src_bow)
    # src_bow_norm = src_bow / torch.sum(src_bow, dim=1, keepdim=True)

    ref_input = (ref_docs, ref_lens, ref_doc_lens, ref_oovs)
    seq2seq_output, topic_model_output = topic_seq2seq_model(src, src_lens, trg, src_oov, max_num_oov, src_mask,
                                                             src_bow_norm,
                                                             ref_input=ref_input,
                                                             graph=graph)

    decoder_dist, h_t, attention_dist, encoder_final_state, coverage = seq2seq_output
    topic_represent, vq_loss, rec_bow = topic_model_output
    ntm_loss = F.binary_cross_entropy(rec_bow, src_bow, size_average=False)
    ntm_loss = ntm_loss + vq_loss
    forward_time = time_since(start_time)

    start_time = time.time()

    if opt.copy_attention:  # Compute the loss using target with oov words
        loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                    opt.coverage_loss)
    else:  # Compute the loss using target without oov words
        loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                    opt.coverage_loss)

    loss_compute_time = time_since(start_time)

    total_trg_tokens = sum(trg_lens)

    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("src")
        print(src)
        print(src_oov)
        print(src_lens)
        print(src_mask)
        print("trg")
        print(trg)
        print(trg_oov)
        print(trg_lens)
        print(trg_mask)
        print("oov list")
        print(oov_lists)
        print("Decoder")
        print(decoder_dist)
        print(h_t)
        print(attention_dist)
        raise ValueError("Loss is NaN")

    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        normalization = src.size(0)
    else:
        raise ValueError('The type of loss normalization is invalid.')

    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    loss += ntm_loss
    # back propagation on the normalized loss
    loss.div(normalization).backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        # 裁剪的decoder和encoder 不包含主题模型
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(topic_seq2seq_model.parameters(), opt.max_grad_norm)

    optimizer.step()

    # construct a statistic object for the loss
    stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time,
                          loss_compute_time=loss_compute_time, backward_time=backward_time)

    return stat


def train_model(latent_Seq2Seq_model, optimizer, train_data_loader, valid_data_loader,
                train_ntm_dataloader, bow_dictionary, opt):
    writer = SummaryWriter(os.path.join(opt.model_path, 'log'))
    logging.info('======================  Start Training  =========================')

    total_batch = 0
    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    joint_train_patience = 1
    global_patience = 5
    num_stop_dropping = 0
    num_stop_dropping_global = 0

    t0 = time.time()

    check_pt_model_path = ""
    print("\nEntering main training for %d epochs" % opt.epochs)
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        latent_Seq2Seq_model.train()

        logging.info("The total num of batches: %d, current learning rate:%.6f" %
                     (len(train_data_loader), optimizer.param_groups[0]['lr']))

        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1
            batch_loss_stat = train_one_batch(batch, latent_Seq2Seq_model, optimizer, opt,
                                                           total_batch)
            report_train_loss_statistics.update(batch_loss_stat)
            total_train_loss_statistics.update(batch_loss_stat)

            if (batch_i + 1) % (len(train_data_loader) // 10) == 0:
                print("Train: %d/%d batches, current avg loss: %.3f" %
                      ((batch_i + 1), len(train_data_loader), batch_loss_stat.xent()))

        current_train_ppl = report_train_loss_statistics.ppl()
        current_train_loss = report_train_loss_statistics.xent()
        writer.add_scalar('Train/total_loss', current_train_loss, epoch)
        # test the model on the validation dataset for one epoch
        valid_loss_stat = evaluate_loss(valid_data_loader, latent_Seq2Seq_model, opt)
        current_valid_loss = valid_loss_stat.xent()
        current_valid_ppl = valid_loss_stat.ppl()
        writer.add_scalar('Train/valid_loss', current_valid_loss, epoch)
        # debug
        if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
            logging.info(
                "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
            exit()

        if current_valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
            print("Valid loss drops")
            sys.stdout.flush()
            best_valid_loss = current_valid_loss
            best_valid_ppl = current_valid_ppl
            num_stop_dropping = 0
            num_stop_dropping_global = 0
            # show topic words
            # latent_Seq2Seq_model.topic_model.print_topic_words(bow_dictionary, os.path.join(opt.model_path,
            #                                                                                 'topwords_e%d.txt' % epoch))
            if epoch >= opt.start_checkpoint_at and epoch > opt.p_seq2seq_e and not opt.save_each_epoch:
                check_pt_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.model-%s' %
                                                   (epoch, current_valid_loss,
                                                    convert_time2str(time.time() - t0)))
                # save model parameters
                torch.save(
                    latent_Seq2Seq_model.state_dict(),
                    open(check_pt_model_path, 'wb')
                )
                logging.info('Saving seq2seq checkpoints to %s' % check_pt_model_path)

        else:
            print("Valid loss does not drop")
            sys.stdout.flush()
            num_stop_dropping += 1
            num_stop_dropping_global += 1
            # decay the learning rate by a factor
            for i, param_group in enumerate(optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = old_lr * opt.learning_rate_decay
                if old_lr - new_lr > EPS:
                    param_group['lr'] = new_lr
                    print("The new learning rate for seq2seq is decayed to %.6f" % new_lr)

        if opt.save_each_epoch:
            check_pt_model_path = os.path.join(opt.model_path, 'e%d.train_loss=%.3f.val_loss=%.3f.model-%s' %
                                               (epoch, current_train_loss, current_valid_loss,
                                                convert_time2str(time.time() - t0)))
            torch.save(  # save model parameters
                latent_Seq2Seq_model.state_dict(),
                open(check_pt_model_path, 'wb')
            )
            logging.info('Saving seq2seq checkpoints to %s' % check_pt_model_path)

        # log loss, ppl, and time

        logging.info('Epoch: %d; Time spent: %.2f' % (epoch, time.time() - t0))
        logging.info(
            'avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                current_train_ppl, current_valid_ppl, best_valid_ppl))
        logging.info(
            'avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                current_train_loss, current_valid_loss, best_valid_loss))

        report_train_ppl.append(current_train_ppl)
        report_valid_ppl.append(current_valid_ppl)
        report_train_loss.append(current_train_loss)
        report_valid_loss.append(current_valid_loss)

        report_train_loss_statistics.clear()

        if not opt.save_each_epoch and num_stop_dropping >= opt.early_stop_tolerance:  # not opt.joint_train or
            logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)

            break

    return check_pt_model_path


def main(opt):
    try:
        start_time = time.time()
        train_data_loader, train_bow_loader, valid_data_loader, valid_bow_loader, \
        word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=True)
        opt.bow_vocab_size = len(bow_dictionary)
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        start_time = time.time()
        # loading model
        topic_seq2seq_model = LatentSeq2SeqModel(opt).to(opt.device)
        optimizer_whole = init_optimizers(model=topic_seq2seq_model, opt=opt)

        train_model(topic_seq2seq_model, optimizer_whole, train_data_loader,
                    valid_data_loader, train_bow_loader, bow_dictionary, opt)

        training_time = time_since(start_time)

        logging.info('Time for training: %.1f' % training_time)

    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.my_own_opts(parser)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    logging = config.init_logging(log_file=opt.model_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
