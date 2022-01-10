# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 下午2:46
# @Author  : WuDiDaBinGe
# @FileName: generate_retriever.py
# @Software: PyCharm
"""
use retriever to generator source retriever
"""
import time
from functools import partial
from multiprocessing import Pool

import torch
from sentence_transformers import SentenceTransformer
import os
import pickle
import faiss
from faiss import normalize_L2
import numpy as np
from retrievers.utils import read_tokenized_src_file
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


class BertRanker(object):
    """
        使用Bert对语料库进行索引，检索最相近的几篇文章
        ref_docs:[]是一个所有文章的列表
    """

    def __init__(self, ref_dir, ref_docs):
        self.ref_docs = ref_docs
        model_name = "allenai-specter"
        model = "/home/yxb/.cache/torch/sentence_transformers/sentence-transformers_allenai-specter"
        if "Weibo" in ref_dir:
            model_name = "paraphrase-multilingual-mpnet-base-v2"
            model = "/home/yxb/.cache/torch/sentence_transformers/sentence-transformers_paraphrase-multilingual-mpnet-base-v2"
        self.model = SentenceTransformer(model)

        embed_cache_path = ref_dir + '/embeddings-{}.pkl'.format(model_name.replace('/', '_'))
        self.index = self.build_index(embed_cache_path)

    def build_index(self, embed_cache_path):
        embedding_size = 768  # Size of embeddings
        n_clusters = 900
        quantizer = faiss.IndexFlatIP(embedding_size)
        index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

        index.nprobe = 200

        # Check if embedding cache path exists
        if not os.path.exists(embed_cache_path):
            corpus_embeddings = self.model.encode(self.ref_docs, show_progress_bar=True, convert_to_numpy=True)

            # Create the FAITS index
            print("Start creating FAISS index")
            # First, we need to normalize vectors to unit length
            normalize_L2(corpus_embeddings)
            # corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            # warn： default token_pattern will ignore singe token
            print("Store corpus_embeddings on disc")
            with open(embed_cache_path, "wb") as fOut:
                pickle.dump({'corpus_embeddings': corpus_embeddings}, fOut)
        else:
            print("Load pre-computed embeddings from disc")
            with open(embed_cache_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                corpus_embeddings = cache_data['corpus_embeddings']
        # Then we train the index to find a suitable clustering
        index.train(corpus_embeddings)
        # Finally we add all embeddings to the index
        index.add(corpus_embeddings)

        return index

    def batch_closest_docs(self, queries, k=1):
        query_embedding = self.model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
        # FAISS works with inner product (dot product). When we normalize vectors to unit length,
        # inner product is equal to cosine similarity
        normalize_L2(query_embedding)
        distances, corpus_ids = self.index.search(query_embedding, k)
        # normalize score to integer between [0, 9]
        distances = np.round(distances * 9)
        distances[distances < 0] = 0
        return corpus_ids


class BM25Ranker(object):
    def __init__(self, ref_docs):
        self.ref_docs = ref_docs
        self.token_crops = [doc.split(" ") for doc in self.ref_docs]
        self.bm25 = BM25Okapi(self.token_crops)

    def batch_closest_docs(self, querys, k=20):
        querys = [q.split(" ") for q in querys]
        t1 = time.time()
        with Pool(10) as processes:
            ref_ids = processes.map(partial(self.get_scores_, k=k), querys)
        t2 = time.time()
        print("Time use:", t2 - t1)
        return ref_ids
        # self.bm25.get_scores()

    def get_scores_(self, query, k=20):
        doc_scores = self.bm25.get_scores(query)
        return doc_scores.argsort()[::-1][:k]


def retriever_text(vocab_path, dataset_dir, mode="train"):
    # load vocab
    word2idx, idx2word, vocab, bow_dictionary = word2idx, idx2word, vocab, bow_dictionary = torch.load(vocab_path)
    retriever = BertRanker(ref_dir=dataset_dir)
    if mode == "train":
        src_docs = os.path.join(dataset_dir, "train_src.txt")
        trg_docs = os.path.join(dataset_dir, "train_trg.txt")
    elif mode == "valid":
        src_docs = os.path.join(dataset_dir, "valid_src.txt")
        trg_docs = os.path.join(dataset_dir, "valid_trg.txt")
    else:
        src_docs = os.path.join(dataset_dir, "test_src.txt")
        trg_docs = os.path.join(dataset_dir, "test_trg.txt")
    docs = [line.strip() for line in open(src_docs, 'r')]
    docs_trg = [line.strip() for line in open(trg_docs, 'r')]

    ref_index = retriever.batch_closest_docs(docs, 10)
    ref_index = ref_index[:, 1:]  # 去掉自身
    ref_docs = ["<sep>".join([docs[ref_id] for ref_id in ref_doc]) for ref_doc in ref_index]
    ref_docs_trg = ["<sep>".join([docs_trg[ref_id] for ref_id in ref_doc]) for ref_doc in ref_index]
    # write src  and refs
    out_path = os.path.join(dataset_dir, "src_refs.txt")
    out_file = open(out_path, 'w')
    for src, ref_src in zip(docs, ref_docs):
        out_file.write(src + '<sep>' + ref_src + '\n')
    out_file.close()
    # src_trg and ref_trgs
    out_path_trg = os.path.join(dataset_dir, "src_refs_trg.txt")
    out_file = open(out_path_trg, "w")
    for src_trg, ref_trg in zip(docs_trg, ref_docs_trg):
        out_file.write(src_trg + '<seq>' + ref_trg + '\n')
    out_file.close()
    return ref_docs


def read_trg_map_posts(trg_map_posts_path):
    trgs_all = []
    posts_all = []
    with open(trg_map_posts_path, "r") as support_file:
        for line in support_file:
            trg_posts_list = line.strip().split("<trg>")
            trg = trg_posts_list[0]
            posts = trg_posts_list[1]
            trgs_all.append(trg)
            posts_all.append(posts)
    return trgs_all, posts_all


def output_reference_keys_by_supports(data_dir, support_dir, mode="train", use_bm25=False):
    trgs_all, posts_all = read_trg_map_posts(os.path.join(support_dir, "trg_map_posts.txt"))
    assert len(trgs_all) == len(posts_all)
    if use_bm25:
        retriever = BM25Ranker(ref_docs=posts_all)
    else:
        retriever = BertRanker(ref_dir=support_dir, ref_docs=posts_all)

    src_docs = os.path.join(data_dir, "{}_src.txt".format(mode))
    trg_docs = os.path.join(data_dir, "{}_trg.txt".format(mode))
    out_path = os.path.join(data_dir, "{}_ret_support.txt".format(mode))

    docs = [line.strip() for line in open(src_docs, 'r')]
    docs_trg = [line.strip() for line in open(trg_docs, 'r')]

    ref_index = retriever.batch_closest_docs(docs, 20)
    ref_keys = [";".join([trgs_all[ref_id] for ref_id in ref_doc]) for ref_doc in ref_index]
    # 写文件
    not_contain_trg = 0
    with open(out_path, "w") as outfile:
        for index, refs in enumerate(ref_keys):
            trg_list = docs_trg[index].split(";")
            outfile.write(refs + '\n')
            # 处理：如果没有检索到 gold keyphrase 的策略
            temp = 0
            for trg in trg_list:
                if trg in refs:
                    temp += 1
            if temp == 0:
                not_contain_trg += 1
    print("Have {} posts not retriver own keyphrase!".format(not_contain_trg))
    print("Have all {} posts".format(len(docs)))


def output_reference_keys_by_single(data_dir, mode="train", use_bm25=False):
    ref_path = os.path.join(data_dir, "train_src.txt")
    ref_trg_path = os.path.join(data_dir, "train_trg.txt")
    ref_docs = read_tokenized_src_file(ref_path)
    ref_keys = [line.strip() for line in open(ref_trg_path, "r")]
    if use_bm25:
        retriever = BM25Ranker(ref_docs=ref_docs)
    else:
        retriever = BertRanker(ref_dir=data_dir, ref_docs=ref_docs)

    src_docs = os.path.join(data_dir, "{}_src.txt".format(mode))
    trg_docs = os.path.join(data_dir, "{}_trg.txt".format(mode))
    out_path = os.path.join(data_dir, "{}_ret.txt".format(mode))

    docs = [line.strip() for line in open(src_docs, 'r')]
    docs_trg = [line.strip() for line in open(trg_docs, 'r')]
    if mode == "train":
        ref_index = retriever.batch_closest_docs(docs, 21)
        ref_index = [ref_index[1:] for ref_index in ref_index]
    else:
        ref_index = retriever.batch_closest_docs(docs, 20)
        ref_index = [ref_index[1:] for ref_index in ref_index]
    rank_ref_keys = [[ref_keys[ref_id] for ref_id in ref_doc] for ref_doc in ref_index]

    # 写文件
    not_contain_trg = 0
    with open(out_path, "w") as outfile:
        for index, refs_list in enumerate(rank_ref_keys):
            trg_list = docs_trg[index].split(";")
            # 处理：如果没有检索到 gold keyphrase 的策略
            temp = 0
            for trg in trg_list:
                if trg in refs_list:
                    temp += 1
            if temp == 0:
                not_contain_trg += 1
            # else:
            #     # 若包含关键词 将它放到首位
            #     # refs_list = trg_list[:temp] "<eos>".join(trg_list[:temp]) + "<eos>" + refs_list
            #     refs_list = refs_list[:1] + trg_list[:temp] + refs_list[1:]
            outfile.write(";".join(refs_list) + '\n')
    print("Have {} posts not retriver own keyphrase!".format(not_contain_trg))
    print("Have all {} posts".format(len(docs)))


if __name__ == '__main__':
    data_name_list = ['Twitter', 'Weibo', 'StackExchange']
    for dataset in data_name_list:
        support_dataset = dataset + "Support"
        dataset_dir = "../data/{}".format(dataset)
        support_dir = "../data/{}".format(support_dataset)
        # retriever_text(vocab_path, dataset_dir)
        output_reference_keys_by_supports(dataset_dir, support_dir, mode="test", use_bm25=True)
        # output_reference_keys_by_single(dataset_dir, use_bm25=True)
        # 900 16940  test: 2081 / 4630
        # 800 16933
        # 500 16925  test: 2252 / 4630

        # BM25算法 test:1523/4630  train12213/37036
        # BM25算法-support 957/4630
