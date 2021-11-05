import torch

from pykp.modules.attention_modules import TopicMemeoryMechanism, ContextTopicAttention

if __name__ == '__main__':
    embed_size = 5
    topic_num = 10
    bow_size = 100
    batch_size = 8
    seq_len = 3
    topic_memory = TopicMemeoryMechanism(embed_size=embed_size,topic_num=topic_num,bow_size=bow_size)
    y_embed = torch.randn(batch_size, embed_size)
    topic_words = torch.randn(topic_num, bow_size)
    topic_represent = torch.randn(batch_size, topic_num)
    out = topic_memory(y_embed, topic_words, topic_represent)
    # print(out)

    contextTopicAttention = ContextTopicAttention(encoder_hidden_size=embed_size, topic_num=topic_num, topic_emb_dim=embed_size)
    encoder_memory = torch.randn(batch_size,seq_len, embed_size)
    attention_dist = torch.randn(batch_size, seq_len)
    topic_emb = torch.randn(topic_num, embed_size)
    topic_dist = torch.randn(batch_size, topic_num)
    topic_words_attention = contextTopicAttention(encoder_memory, attention_dist, topic_emb, topic_dist)
    print(topic_words_attention[1].shape)