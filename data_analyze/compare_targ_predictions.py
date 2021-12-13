import pandas as pd
target_file_path = "../data/Weibo/test_trg.txt"
sample_file_path = "../my_sample_prediction/Weibo_predictions.txt"
source_post_path = "../data/Weibo/test_src.txt"
def out_compare_result(predict_file_path,target_path=target_file_path, sample_path=sample_file_path, source_path=source_post_path, top_k=3):
    # 读取原帖
    source_file = open(source_path, 'r', encoding='utf-8')
    source_posts = [line.strip() for line in source_file]
    source_file.close()


    target_file = open(target_path, 'r', encoding='utf-8')
    target_words = [line.strip().split(";") for line in target_file]
    target_file.close()

    sample_file = open(sample_path, 'r', encoding='utf-8')
    sample_top_words = [line.strip().split(";")[:top_k] for line in sample_file]
    sample_file.close()

    predict_file =  open(predict_file_path, "r", encoding='utf-8')
    predict_top_words = [line.strip().split(";")[:top_k] for line in predict_file]
    predict_file.close()

    # 生成对比文件
    assert len(target_words)==len(sample_top_words)
    assert len(target_words)==len(predict_top_words)
    result = {
        'post':[],
        'target_keys':[],
        'predict_keys':[],
        'predict_correct_num': [],
        'sample_keys':[],
        'sample_correct_num':[]
    }
    for index in range(len(target_words)):
        predict_correct_num = 0
        sample_correct_num = 0
        for k_index, keyphrase in enumerate(target_words[index]):
            if keyphrase in predict_top_words[index]:
                predict_correct_num = predict_correct_num + 1
            if keyphrase in sample_top_words[index]:
                sample_correct_num = sample_correct_num + 1
        target_keys = ";".join(target_words[index])
        predict_keys = ";".join(predict_top_words[index])
        sample_keys = ";".join(sample_top_words[index])
        result['target_keys'].append(target_keys)
        result['predict_keys'].append(predict_keys)
        result['sample_keys'].append(sample_keys)
        result['predict_correct_num'].append(predict_correct_num)
        result['sample_correct_num'].append(sample_correct_num)
        result['post'].append(source_posts[index])

    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values(by=['predict_correct_num','sample_correct_num'])
    result_df.to_csv("./compare_result.csv", index=False)

if __name__ == '__main__':
    predict_file_path = "E:\MyCoding\TAKG\pred\predict__Weibo_s100_t10.joint_train.add_two_loss.p_1_iterate.use_topic.topic_num50.topic_attn.topic_copy.topic_attn_in.ntm_warm_up_0.copy.seed9527.emb150.vs50000.dec300.20211102-140536__e104.val_loss=1.290.model-0h-23m\predictions.txt"
    out_compare_result(predict_file_path)
