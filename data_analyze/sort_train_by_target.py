# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 上午11:02
# @Author  : WuDiDaBinGe
# @FileName: sort_train_by_target.py
# @Software: PyCharm
from collections import Counter
import random
from utils.string_helper import stem_word_list, stem_word_list


def count_key_not_in_trg_doc(trg_src_dict):
    """
        统计 关键词不在其标记的所有文档中出现
    """
    absent_key = []
    max_len_posts = -1
    key_posts_num_b10 = 0
    for key, posts in trg_src_dict.items():
        all_docs = " ".join(posts)
        if key not in all_docs:
            # print(key + '\t' + str(len(posts)))
            absent_key.append(key)
        max_len_posts = max(max_len_posts, len(posts))
        if len(posts) < 2:
            key_posts_num_b10 += 1
    return {
        'absent_key_list': absent_key,
        'max_len_posts': max_len_posts,
        'key_posts_num_b10': key_posts_num_b10
    }


def count_absent_not_in_crops(trg_src_dict, src_list):
    # 统计关键词在不在整个训练集中出现
    # 关键词的集合
    absent_key_all = []
    trg_set = set(trg_src_dict.keys())
    for keyphrase in trg_set:
        flag = False
        for src_post in src_list:
            if keyphrase in src_post:
                flag = True
                break
        if not flag:
            absent_key_all.append(keyphrase)
            print(keyphrase + '\t' + str(len(trg_src_dict[keyphrase])))
    print("The key not appear in train set:{}".format(len(absent_key_all)))
    return absent_key_all


def output_sorted_target(train_src_path, train_trg_path, out_file_path):
    # 每一个元素为一个tuple (关键词：帖子)
    src_trg_list = []
    # 文章集合
    src_list = []
    # key: 关键词 value:[]列表 保存关键词下的帖子
    trg_src_dict = {}
    absent_key_post = 0

    for src_line, trg_line in zip(open(train_src_path, 'r'), open(train_trg_path, 'r')):
        src_token_list = src_line.strip().split(" ")
        src_stream_list_ = stem_word_list(src_token_list)
        src_stream_str = " ".join(src_stream_list_)

        src_list.append(src_stream_str)
        trg_list = trg_line.strip().split(';')
        for trg in trg_list:
            trg_stream_list = stem_word_list(trg.strip().split(" "))
            trg_stream_str = " ".join(trg_stream_list)
            src_trg_list.append((trg_stream_str, src_stream_str))
            if trg_stream_str not in src_stream_str:
                absent_key_post += 1
            if trg_stream_str not in trg_src_dict.keys():
                trg_src_dict[trg_stream_str] = []
            trg_src_dict[trg_stream_str].append(src_stream_str.strip())
    # 根据关键词排序
    src_trg_list = sorted(src_trg_list,
                          key=lambda example: (len(trg_src_dict[example[0]]), example[0], len(example[0])))
    # write in file
    out_file = open(out_file_path, 'w')
    for trg, src in src_trg_list:
        out_file.write('{}\t\t{}\n'.format(trg, src))
    out_file.close()
    result = count_key_not_in_trg_doc(trg_src_dict)
    print("absent_key doc:{}".format(absent_key_post))
    print("absent_key not in train:{}".format(len(result['absent_key_list'])))
    print("The key contain less 10 pots:{}".format(result['key_posts_num_b10']))
    print("The key contain max num pots:{}".format(result['max_len_posts']))
    count_absent_not_in_crops(trg_src_dict, src_list)


def merge_line_by_line(train_src_file, trg_src_file, merge_file):
    src_file = open(train_src_file, "r")
    trg_file = open(trg_src_file, "r")
    doc_trg = []
    for src_line, trg_line in zip(src_file, trg_file):
        line = src_line.strip() + " <sep> " + trg_line.strip().replace(";", " ")
        doc_trg.append(line)
    src_file.close()
    trg_file.close()

    out_file = open(merge_file, "w")
    out_file.write('\n'.join(doc_trg))
    out_file.close()


def get_key_not_in_train(trg_file1, trg_file2):
    train_key_counter = Counter()
    for train_trg in open(trg_file1, 'r'):
        trg_list = train_trg.strip().split(';')
        train_key_counter.update(trg_list)
    total_num = 0
    absent_key = []
    for test_trg in open(trg_file2, 'r'):
        trg_list = test_trg.strip().split(';')
        for trg in trg_list:
            if trg not in train_key_counter.keys():
                absent_key.append(trg)
        total_num += 1
    print("\n".join(absent_key))
    print("The oov keyphrase's num is {}, rate is {}".format(len(absent_key), len(absent_key) / total_num))


def output_keyphrase_map_posts(train_src_path, train_trg_path, out_file_path):
    # key: 关键词 value:[]列表 保存关键词下的帖子
    trg_src_dict = {}
    for src_line, trg_line in zip(open(train_src_path, 'r'), open(train_trg_path, 'r')):
        src_line = src_line.strip()

        trg_list = trg_line.strip().split(';')
        for trg in trg_list:
            if trg not in trg_src_dict.keys():
                trg_src_dict[trg] = []
            trg_src_dict[trg].append(src_line)

    def get_posts_score_to_key(keyphrase, post):
        """
        统计帖子包含关键词token的个数
        """
        trg_tokens = keyphrase.strip().split(" ")
        trg_tokens = stem_word_list(trg_tokens)

        post_tokens = post.strip().split(" ")
        post_tokens = stem_word_list(post_tokens)
        res = 0
        for token in trg_tokens:
            if token in post_tokens:
                res += 1
        return res

    # sort and shuffle value list
    trg_src_dict = {k: v for k, v in sorted(trg_src_dict.items(), key=lambda example: len(example[1]))}
    out_file = open(out_file_path, "w")
    for key, posts in trg_src_dict.items():
        # 包含关键词的token多并且帖子长的排在前面
        # posts = sorted(posts, key=lambda post: (get_posts_score_to_key(key, post), len(post)), reverse=True)
        random.shuffle(posts)
        posts = " ".join(posts)
        out_file.write(key + "<trg> " + posts + '\n')
    out_file.close()


if __name__ == '__main__':
    data_name_list = ['Twitter', 'Weibo', 'StackExchange']
    for dataset_name in data_name_list:
        train_src = '../data/' + dataset_name + '/train_src.txt'
        train_trg = '../data/' + dataset_name + '/train_trg.txt'

        valid_trg = '../data/' + dataset_name + '/valid_trg.txt'
        test_trg = '../data/' + dataset_name + '/test_trg.txt'

        out_path = '../data/' + dataset_name + '/sort_post_trg.txt'
        out_path = '../data/' + dataset_name + '/trg_map_posts.txt'
        # output_sorted_target(train_src, train_trg, out_path)
        # merge_line_by_line(train_src, train_trg, out_path)
        # get_key_not_in_train(train_trg, test_trg)
        output_keyphrase_map_posts(train_src, train_trg, out_path)
