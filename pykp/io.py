# -*- coding: utf-8 -*-
"""
Python File Template
Built on the source code of seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""
import logging
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.utils.data

from utils.functions import pad

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
SEP_WORD = '<sep>'
DIGIT = '<digit>'


# TITLE_ABS_SEP = '[SEP]'
# PEOS_WORD = '<peos>'

class KeyphraseDataset(torch.utils.data.Dataset):
    def __init__(self, examples, word2idx, idx2word, bow_dictionary,
                 type='one2one', delimiter_type=0, load_train=True, remove_src_eos=False, use_multidoc_graph=True,
                 use_multidoc_copy=True):
        # keys of matter. `src_oov` is for mapping pointed word to dict,
        # `oov_dict` is for determining the dim of predicted logit: dim=vocab_size+max_oov_dict_in_batch
        assert type in ['one2one', 'one2many']
        keys = ['src', 'trg', 'trg_copy', 'src_oov', 'oov_dict', 'oov_list', 'src_str', 'trg_str', 'src_bow']
        if use_multidoc_graph:
            keys += ['ref_docs']
            keys += ['graph']
            keys += ['query_embedding']
            if use_multidoc_copy:
                keys += ['ref_oov']
        filtered_examples = []

        for e in examples:
            filtered_example = {}
            for k in keys:
                filtered_example[k] = e[k]
            if 'oov_list' in filtered_example:
                filtered_example['oov_number'] = len(filtered_example['oov_list'])

            filtered_examples.append(filtered_example)

        self.examples = filtered_examples
        self.word2idx = word2idx
        self.id2xword = idx2word
        self.bow_dictionary = bow_dictionary
        self.pad_idx = word2idx[PAD_WORD]
        self.type = type
        if delimiter_type == 0:
            self.delimiter = self.word2idx[SEP_WORD]
        else:
            self.delimiter = self.word2idx[EOS_WORD]
        self.load_train = load_train
        self.remove_src_eos = remove_src_eos
        self.use_multidoc_graph = use_multidoc_graph
        self.use_multidoc_copy = use_multidoc_copy

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, input_list):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = self.pad_idx * np.ones((len(input_list), max_seq_len))

        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]

        padded_batch = torch.LongTensor(padded_batch)

        input_mask = torch.ne(padded_batch, self.pad_idx)
        input_mask = input_mask.type(torch.FloatTensor)

        return padded_batch, input_list_lens, input_mask

    def _pad_bow(self, input_list):
        bow_vocab = len(self.bow_dictionary)
        res_src_bow = np.zeros((len(input_list), bow_vocab))
        for idx, bow in enumerate(input_list):
            bow_k = [k for k, v in bow]
            bow_v = [v for k, v in bow]
            res_src_bow[idx, bow_k] = bow_v

        return torch.FloatTensor(res_src_bow)

    def collate_bow(self, batches):
        if self.remove_src_eos:
            src = [b['src'] for b in batches]
        else:
            src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
        src, src_lens, src_mask = self._pad(src)
        src_bow = [b['src_bow'] for b in batches]
        # ref docs
        if self.use_multidoc_graph:
            ref_docs = [b['ref_docs'] for b in batches]
            from retrievers.utils import build_graph
            graph = [build_graph(**b['graph']) for b in batches]
            ref_docs, ref_lens, ref_doc_lens = self._pad2d(ref_docs)

        else:
            ref_docs, graph, ref_lens, ref_doc_lens = None, None, None, None
        return src, src_lens, self._pad_bow(src_bow), ref_docs, ref_lens, ref_doc_lens, graph

    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        assert self.type == 'one2one', 'The type of dataset should be one2one.'
        if self.remove_src_eos:
            # source with oov words replaced by <unk>
            src = [b['src'] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] for b in batches]
        else:
            # source with oov words replaced by <unk>
            src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]

        # target_input: input to decoder, ends with <eos> and oovs are replaced with <unk>
        trg = [b['trg'] + [self.word2idx[EOS_WORD]] for b in batches]

        # target for copy model, ends with <eos>, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_oov = [b['trg_copy'] + [self.word2idx[EOS_WORD]] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]
        src_bow = [b['src_bow'] for b in batches]
        # ref docs
        if self.use_multidoc_graph:
            # query_embeddings = [b['query_embedding'] for b in batches]
            # query_embeddings = torch.Tensor(query_embeddings)
            ref_docs = [b['ref_docs'] for b in batches]
            if self.use_multidoc_copy:
                assert batches[0]['ref_oov'] is not None, "set use_multidoc_copy in preprocess!"
                ref_oovs = [b['ref_oov'] for b in batches]
            else:
                ref_oovs = None
            from retrievers.utils import build_graph
            graph = [build_graph(**b['graph']) for b in batches]

            ref_docs, ref_lens, ref_doc_lens = self._pad2d(ref_docs)
            if self.use_multidoc_copy:
                ref_oovs, _, _ = self._pad(ref_oovs)
                #  covert size to [batch, max_len1, max_len2]
                ref_oovs = pad(ref_oovs, ref_doc_lens)
        else:
            ref_docs, ref_oovs, graph, ref_lens, ref_doc_lens, query_embeddings = None, None, None, None, None, None

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        # seq_pairs = sorted(zip(src, trg, trg_oov, src_oov, oov_lists, src_bow), key=lambda p: len(p[0]), reverse=True)
        # src, trg, trg_oov, src_oov, oov_lists, src_bow = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        trg, trg_lens, trg_mask = self._pad(trg)
        # trg_target, _, _ = self._pad(trg_target)
        trg_oov, _, _ = self._pad(trg_oov)
        src_oov, _, _ = self._pad(src_oov)
        src_bow = self._pad_bow(src_bow)
        query_embeddings = None
        return src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, src_bow, \
               ref_docs, ref_lens, ref_doc_lens, ref_oovs, graph, query_embeddings

    def _pad2d(self, input_list):
        """
         list of list of 1d int:
         pad for ref docs
        """
        batch_size = len(input_list)
        # 每一个文档有多少相关文档
        input_list_lens1 = [len(l) for l in input_list]

        max_seq_len1 = max(input_list_lens1)
        # [batch_size, max_seq_len1]
        input_list_lens2 = self.word2idx[PAD_WORD] * np.ones((batch_size, max_seq_len1))

        max_seq_len = max([max([len(i) for i in l]) for l in input_list])
        padded_batch = self.word2idx[PAD_WORD] * np.ones((batch_size, max_seq_len1, max_seq_len))

        for i in range(batch_size):
            for j in range(len(input_list[i])):
                current_len = len(input_list[i][j])
                input_list_lens2[i][j] = current_len
                padded_batch[i][j][:current_len] = input_list[i][j]

        padded_batch = torch.LongTensor(padded_batch)
        input_list_lens1 = torch.LongTensor(input_list_lens1)
        input_list_lens2 = torch.LongTensor(input_list_lens2)

        return padded_batch, input_list_lens1, input_list_lens2

    def collate_fn_one2many(self, batches):
        assert self.type == 'one2many', 'The type of dataset should be one2many.'
        if self.remove_src_eos:
            # source with oov words replaced by <unk>
            src = [b['src'] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] for b in batches]
        else:
            # source with oov words replaced by <unk>
            src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]

        batch_size = len(src)

        # trg: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oov replaced by UNK
        # trg_oov: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        if self.load_train:
            trg = []
            trg_oov = []
            for b in batches:
                trg_concat = []
                trg_oov_concat = []
                trg_size = len(b['trg'])
                assert len(b['trg']) == len(b['trg_copy'])
                for trg_idx, (trg_phase, trg_phase_oov) in enumerate(zip(b['trg'], b[
                    'trg_copy'])):  # b['trg'] contains a list of targets, each target is a list of indices
                    # for trg_idx, a in enumerate(zip(b['trg'], b['trg_copy'])):
                    # trg_phase, trg_phase_oov = a
                    if trg_idx == trg_size - 1:  # if this is the last keyphrase, end with <eos>
                        trg_concat += trg_phase + [self.word2idx[EOS_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[EOS_WORD]]
                    else:
                        trg_concat += trg_phase + [
                            self.delimiter]  # trg_concat = [target_1] + [delimiter] + [target_2] + [delimiter] + ...
                        trg_oov_concat += trg_phase_oov + [self.delimiter]
                trg.append(trg_concat)
                trg_oov.append(trg_oov_concat)
        else:
            trg, trg_oov = None, None
        # trg = [[t + [self.word2idx[EOS_WORD]] for t in b['trg']] for b in batches]
        # trg_oov = [[t + [self.word2idx[EOS_WORD]] for t in b['trg_copy']] for b in batches]
        # ref docs

        oov_lists = [b['oov_list'] for b in batches]
        src_bow = [b['src_bow'] for b in batches]

        # b['src_str'] is a word_list for source text, b['trg_str'] is a list of word list
        src_str = [b['src_str'] for b in batches]
        trg_str = [b['trg_str'] for b in batches]

        original_indices = list(range(batch_size))
        # get ref oovs
        if self.use_multidoc_graph:
            # query_embeddings = [b['query_embedding'] for b in batches]
            # query_embeddings = torch.Tensor(query_embeddings)
            ref_docs = [b['ref_docs'] for b in batches]
            if self.use_multidoc_copy:
                assert batches[0]['ref_oov'] is not None, "set use_multidoc_copy in preprocess!"
                ref_oovs = [b['ref_oov'] for b in batches]
            else:
                ref_oovs = None
            from retrievers.utils import build_graph
            graph = [build_graph(**b['graph']) for b in batches]

            ref_docs, ref_lens, ref_doc_lens = self._pad2d(ref_docs)
            if self.use_multidoc_copy:
                ref_oovs, _, _ = self._pad(ref_oovs)
                #  covert size to [batch, max_len1, max_len2]
                ref_oovs = pad(ref_oovs, ref_doc_lens)
        else:
            ref_docs, ref_oovs, graph, ref_lens, ref_doc_lens, query_embeddings = None, None, None, None, None, None

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        # if self.load_train:
        #     seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices),
        #                        key=lambda p: len(p[0]), reverse=True)
        #     src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices = zip(*seq_pairs)
        # else:
        #     seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, original_indices, src_bow),
        #                        key=lambda p: len(p[0]), reverse=True)
        #     src, src_oov, oov_lists, src_str, trg_str, original_indices, src_bow = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        src_oov, _, _ = self._pad(src_oov)
        if self.load_train:
            trg, trg_lens, trg_mask = self._pad(trg)
            trg_oov, _, _ = self._pad(trg_oov)
        else:
            trg_lens, trg_mask = None, None

        src_bow = self._pad_bow(src_bow)
        query_embeddings = None
        return src, src_lens, src_mask, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, trg_lens, trg_mask, original_indices, src_bow, \
               ref_docs, ref_lens, ref_doc_lens, ref_oovs, graph, query_embeddings


def build_dataset(src_trgs_pairs, word2idx, bow_dictionary, opt, mode='one2one', include_original=True, is_train=True, tfidf_model=None):
    """
    Standard process for copy model
    :param mode: one2one or one2many
    :param include_original: keep the original texts of source and target
    :return:
    """
    _build_one_example = partial(build_one_example, opt=opt, mode=mode, include_original=include_original,
                                 is_train=is_train, word2idx=word2idx, bow_dictionary=bow_dictionary, tf_idf_model=tfidf_model)

    if not opt.retriever or not opt.dense_retrieve:
        with Pool(opt.num_workers) as processes:
            examples = processes.starmap(_build_one_example, zip(src_trgs_pairs))
    # retriever is not None ,dense_retrieve is true
    else:
        ref_docs_tokenized, graph_utils, query_embedding = opt.retriever.batch_maybe_retrieving_building_graph(
            [' '.join(pair[0]) for pair in src_trgs_pairs], word2idx,
            vocab_size=opt.vocab_size, is_train=is_train)
        if graph_utils is None:
            graph_utils = [None] * len(src_trgs_pairs)
        examples = [_build_one_example(i, j, k, q) for i, j, k, q in zip(src_trgs_pairs, ref_docs_tokenized, graph_utils, query_embedding)]

    if mode == 'one2one':
        return_examples = []
        for exps in examples:
            for ex in exps:
                return_examples.append(ex)
    else:
        return_examples = examples

    return return_examples


def build_one_example(src_tgt_pair, ref_docs_tokenized=None, graph_utils=None, query_embedding=None, opt=None, mode='one2one',
                      include_original=False, is_train=True, word2idx=None, bow_dictionary=None, tf_idf_model=None):
    assert word2idx is not None, "word2idx should not be None"
    assert bow_dictionary is not None, "bow_dictionary should not be None"
    source, targets = src_tgt_pair
    src = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size
           else word2idx[UNK_WORD] for w in source]
    src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2idx, opt.vocab_size, opt.max_unk_words)
    # preprocess ref posts
    if opt.retriever is not None:
        if ref_docs_tokenized is None:
            ref_docs_tokenized, graph_utils = opt.retriever.maybe_retrieving_building_graph(source, word2idx,
                                                                                            vocab_size=len(
                                                                                                word2idx),
                                                                                            is_train=is_train)
        ref_docs = [
            [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in doc]
            for doc in ref_docs_tokenized]
        # print("Source post is:{}".format(str(source)))
        # print("Ref post is:{}".format(str(ref_docs_tokenized)))

        if opt.use_multidoc_copy:
            flattened_ref = []
            for ref in ref_docs_tokenized:
                flattened_ref += ref
            ref_oov, oov_dict, oov_list = extend_vocab_OOV(flattened_ref, word2idx, opt.vocab_size,
                                                           opt.max_unk_words, pre_oov_dict=oov_dict)
        else:
            ref_oov = None
    else:
        ref_docs, ref_oov, graph_utils = None, None, None

    examples = []  # for one-to-many
    for target in targets:
        example = {}
        if opt.retriever is not None:
            example['ref_docs'] = ref_docs
            if graph_utils:
                example['graph'] = graph_utils
            if opt.use_multidoc_copy:
                # the doc represation using pretrained model
                example['query_embedding'] = query_embedding
                example['ref_oov'] = ref_oov
        if include_original:
            example['src_str'] = source
            example['trg_str'] = target

        example['src'] = src
        example['src_bow'] = bow_dictionary.doc2bow(source)
        if tf_idf_model is not None:
            example['src_bow'] = tf_idf_model[example['src_bow']]
        if len(example['src_bow']) == 0:
            # for train and valid data, we do not account for zero bow, contrary for test
            if mode == "one2one":
                continue
            # print("%d pairs have zero bow" % idx)

        trg = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size
               else word2idx[UNK_WORD] for w in target]

        example['trg'] = trg
        example['src_oov'] = src_oov
        example['oov_dict'] = oov_dict
        example['oov_list'] = oov_list

        # oov words are replaced with new index
        trg_copy = []
        for w in target:
            if w in word2idx and word2idx[w] < opt.vocab_size:
                trg_copy.append(word2idx[w])
            elif w in oov_dict:
                trg_copy.append(oov_dict[w])
            else:
                trg_copy.append(word2idx[UNK_WORD])

        example['trg_copy'] = trg_copy
        examples.append(example)

    if mode == 'one2many' and len(examples) > 0:
        o2m_example = {}
        keys = examples[0].keys()
        for key in keys:
            if key.startswith('src') or key.startswith('oov') \
                    or key.startswith('ref') or key.startswith('graph') or key.startswith('query'):
                o2m_example[key] = examples[0][key]
            else:
                o2m_example[key] = [e[key] for e in examples]
        if include_original:
            assert len(o2m_example['src']) == len(o2m_example['src_oov']) == len(o2m_example['src_str'])
            assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
            assert len(o2m_example['trg']) == len(o2m_example['trg_copy']) == len(o2m_example['trg_str'])
        else:
            assert len(o2m_example['src']) == len(o2m_example['src_oov'])
            assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
            assert len(o2m_example['trg']) == len(o2m_example['trg_copy'])
        return o2m_example
    else:
        return examples


def extend_vocab_OOV(source_words, word2idx, vocab_size, max_unk_words, pre_oov_dict=None):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    WARNING: if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
    Args:
        source_words: list of words (strings)
        word2idx: vocab word2idx
        vocab_size: the maximum acceptable index of word in vocab
    Returns:
        ids: A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs: A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers.
    """
    src_oov = []
    if pre_oov_dict is None:
        oov_dict = {}
    else:
        oov_dict = pre_oov_dict
    for w in source_words:
        if w in word2idx and word2idx[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
            src_oov.append(word2idx[w])
        # 不在规定的词表中
        else:
            if len(oov_dict) < max_unk_words:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                word_id = oov_dict.get(w, len(oov_dict) + vocab_size)
                oov_dict[w] = word_id
                src_oov.append(word_id)
            else:
                # exceeds the maximum number of acceptable oov words, replace it with <unk>
                word_id = word2idx[UNK_WORD]
                src_oov.append(word_id)

    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x: x[1])]
    return src_oov, oov_dict, oov_list
