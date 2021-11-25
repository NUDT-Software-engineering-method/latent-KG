# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 上午11:02
# @Author  : WuDiDaBinGe
# @FileName: sort_train_by_target.py
# @Software: PyCharm
from utils.string_helper import stem_word_list, stem_word_list

def output_sorted_traget(train_src, train_trg, out_path):
    # 每一个元素为一个tuple (关键词：帖子)
    src_trg_list = []
    # 文章集合
    src_list = []
    # key: 关键词 value:[]列表 保存关键词下的帖子
    trg_src_dict = {}
    absent_key_post = 0
    absent_key_crops = 0
    for src_line, trg_line in zip(open(train_src, 'r'), open(train_trg, 'r')):
        src_token_list = src_line.strip().split(" ")
        src_stream_list_ = stem_word_list(src_token_list)
        src_stream_str = " ".join(src_stream_list_)

        src_trg_list.append((trg_line.strip(), src_stream_str))

        src_list.append(src_stream_str)
        trg_list = trg_line.strip().split(';')
        for trg in trg_list:
            trg_stream_list = stem_word_list(trg.strip().split(" "))
            trg_stream_str = " ".join(trg_stream_list)
            if trg_stream_str not in src_stream_str:
                absent_key_post += 1
            if trg_stream_str not in trg_src_dict.keys():
                trg_src_dict[trg_stream_str] = []
            trg_src_dict[trg_stream_str].append(src_stream_str.strip())
    # 根据关键词排序
    src_trg_list = sorted(src_trg_list, key=lambda example: (len(example[0]), example[0]))
    # write in file
    out_file = open(out_path, 'w')
    for trg, src in src_trg_list:
        out_file.write('{}\t\t{}\n'.format(trg, src))
    out_file.close()
    key_posts_num_b10 = 0
    max_len_posts = -1
    # 统计 关键词在不在 列表中
    for key, posts in trg_src_dict.items():
        flag = False
        for post in posts:
            if key in post:
                flag = True
                break
        if not flag:
            absent_key_crops += 1
            #print(key + '\t' + str(len(posts)))
        max_len_posts = max(max_len_posts, len(posts))
        if len(posts) < 10:
            key_posts_num_b10 += 1
    print("absent_key doc:{}".format(absent_key_post))
    print("absent_key not in train:{}".format(absent_key_crops))
    print("The key contain less 10 pots:{}".format(key_posts_num_b10))
    print("The key contain max num pots:{}".format(max_len_posts))
    # 统计关键词在不在训练集中出现
    # 关键词的集合
    absent_key_all_crops = 0
    trg_set = set(trg_src_dict.keys())
    for keyphrase in trg_set:
        flag = False
        for src_post in src_list:
            if keyphrase in src_post:
                flag = True
                break
        if not flag:
            absent_key_all_crops += 1
            print(keyphrase + '\t')
    print("The key not appear in train set:{}".format(absent_key_all_crops))



if __name__ == '__main__':
    dataset_name = 'Twitter'
    train_src = '../data/' + dataset_name + '/train_src.txt'
    train_trg = '../data/' + dataset_name + '/train_trg.txt'
    out_path = '../data/' + dataset_name + '/sorted_trg.txt'
    output_sorted_traget(train_src, train_trg, out_path)
