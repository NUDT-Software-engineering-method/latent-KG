# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 下午2:46
# @Author  : WuDiDaBinGe
# @FileName: generate_retriever.py
# @Software: PyCharm
"""
use retriever to generator source retriever
"""
import torch
from sentence_transformers import SentenceTransformer
import os
import pickle
import faiss
from faiss import normalize_L2
import numpy as np
from retrievers.utils import read_tokenized_src_file
from sklearn.feature_extraction.text import TfidfVectorizer


class BertRanker(object):
    """
    使用Bert对语料库进行索引，检索最相近的几篇文章
    """

    def __init__(self, word2idx, ref_dir):
        self.word2idx = word2idx
        self.ref_dir = ref_dir
        self.ref_path = os.path.join(ref_dir, "train_src.txt")
        model_name = "allenai-specter"
        # model_name = "paraphrase-multilingual-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)
        embed_cache_path = ref_dir + '/embeddings-{}.pkl'.format(model_name.replace('/', '_'))
        self.index, self.tfidf_vectorizer = self.build_index(embed_cache_path)

    def build_index(self, embed_cache_path):
        embedding_size = 768  # Size of embeddings
        n_clusters = 2000
        quantizer = faiss.IndexFlatIP(embedding_size)
        index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

        index.nprobe = 200

        # Check if embedding cache path exists
        if not os.path.exists(embed_cache_path):
            ref_docs = read_tokenized_src_file(self.ref_path)
            corpus_embeddings = self.model.encode(ref_docs, show_progress_bar=True, convert_to_numpy=True)

            # Create the FAITS index
            print("Start creating FAISS index")
            # First, we need to normalize vectors to unit length
            normalize_L2(corpus_embeddings)
            # corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            # warn： default token_pattern will ignore singe token
            tfidf_vectorizer = TfidfVectorizer(tokenizer=str.split,
                                               vocabulary={w: i for w, i in self.word2idx.items()
                                                           if i < len(self.word2idx)})
            tfidf_vectorizer = tfidf_vectorizer.fit(ref_docs)

            id2word = {}
            for w, id in tfidf_vectorizer.vocabulary_.items():
                id2word[id] = w
            tfidf_vectorizer.id2word = id2word

            print("Store corpus_embeddings on disc")
            with open(embed_cache_path, "wb") as fOut:
                pickle.dump({'corpus_embeddings': corpus_embeddings,
                             'tfidf_vectorizer': tfidf_vectorizer}, fOut)
        else:
            print("Load pre-computed embeddings from disc")
            with open(embed_cache_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                corpus_embeddings = cache_data['corpus_embeddings']
                tfidf_vectorizer = cache_data['tfidf_vectorizer']
        # Then we train the index to find a suitable clustering
        index.train(corpus_embeddings)
        # Finally we add all embeddings to the index
        index.add(corpus_embeddings)

        return index, tfidf_vectorizer

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


def retriever_text(vocab_path, dataset_dir, mode="train"):
    # load vocab
    word2idx, idx2word, vocab, bow_dictionary = word2idx, idx2word, vocab, bow_dictionary = torch.load(vocab_path)
    retriever = BertRanker(word2idx=word2idx, ref_dir=dataset_dir)
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
    ref_index = ref_index[:, 1:]
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


if __name__ == '__main__':
    dataset = "Twitter"
    vocab_path = f"../processed_data/{dataset}_s100_t10/vocab.pt".format(dataset=dataset)
    dataset_dir = f"../data/{dataset}".format(dataset=dataset)
    retriever_text(vocab_path, dataset_dir)
