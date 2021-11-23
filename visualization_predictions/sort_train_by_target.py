# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 上午11:02
# @Author  : WuDiDaBinGe
# @FileName: sort_train_by_target.py
# @Software: PyCharm

def output_sorted_traget(train_src, train_trg, out_path):
    src_trg_list = []
    trg_src_dict = {}
    absent_key_post = 0
    absent_key_crops = 0
    for src_line, trg_line in zip(open(train_src, 'r'), open(train_trg, 'r')):
        src_trg_list.append((trg_line.strip(), src_line.strip()))
        trg_list = trg_line.strip().split(';')
        for trg in trg_list:
            if trg not in src_line:
                absent_key_post += 1
            if trg not in trg_src_dict.keys():
                trg_src_dict[trg] = []
            trg_src_dict[trg].append(src_line.strip())
    src_trg_list = sorted(src_trg_list, key=lambda example: (len(example[0]), example[0]))
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
            print(key + '\t' + str(len(posts)))
        max_len_posts = max(max_len_posts, len(posts))
        if len(posts) < 10:
            key_posts_num_b10 += 1
    print("absent_key doc:{}".format(absent_key_post))
    print("absent_key not in train:{}".format(absent_key_crops))
    print("The key contain less 10 pots:{}".format(key_posts_num_b10))
    print("The key contain max num pots:{}".format(max_len_posts))


if __name__ == '__main__':
    dataset_name = 'Weibo'
    train_src = '../data/' + dataset_name + '/train_src.txt'
    train_trg = '../data/' + dataset_name + '/train_trg.txt'
    out_path = '../data/' + dataset_name + '/sorted_trg.txt'
    output_sorted_traget(train_src, train_trg, out_path)
