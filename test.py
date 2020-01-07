import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import jieba
import numpy as np
from gensim import corpora
from gensim import models

torch.manual_seed(1)


# def prepare_sequence(seq, to_ix, len):
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs, dtype=torch.long)


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# 读取文件
training_data = pd.read_csv('task1/train.csv')
# 将文本集生成分词列表
cw = lambda x: jieba.lcut(x)  # 定义分词函数 ，lambda的作用是用来构造函数，lambda 参数：表达式。
texts = training_data.text.apply(cw)
print('文本集：', texts)
# 基于文本集建立词典
dictionary = corpora.Dictionary(texts)
print('词典：', dictionary)
# 提取词典特征数
feature_cnt = len(dictionary.token2id)
print('词典特征数：%d' % feature_cnt)
# 基于词典建立新的【语料库】
corpus = [dictionary.doc2bow(text) for text in texts]
print('语料库：', corpus)
# 用语料库来训练TF-IDF模型
tfidf = models.TfidfModel(corpus)
corpus_vec = tfidf[corpus]
# print(corpus_vec[0])
# print(corpus_tfidf[1])


# # 处理测试集
# # 读取文件
# training_data = pd.read_csv('task1/test_stage1.csv')
# # 将文本集生成分词列表
# cw = lambda x: jieba.lcut(x)  # 定义分词函数 ，lambda的作用是用来构造函数，lambda 参数：表达式。
# texts = training_data.text.apply(cw)
# print('文本集：', texts)
# # 基于测试集建立词典
# test_dictionary = corpora.Dictionary(texts)
# print('词典：', test_dictionary)
# # 提取词典特征数
# test_feature_cnt = len(test_dictionary.token2id)
# print('词典特征数：%d' % test_feature_cnt)
# # 基于词典建立新的【语料库】
# test_corpus = [test_dictionary.doc2bow(text) for text in texts]
# print('语料库：', test_corpus)
# # 用测试集语料库来训练TF-IDF模型
# test_tfidf = models.TfidfModel(test_corpus)
# corpus_vec = test_tfidf[test_corpus]
# print(corpus_vec[0])
#
# # save测试集为矩阵
# c = np.array(corpus_vec)
# df_result = pd.DataFrame(c.T, columns=['text'])
# df_result.to_csv('corpus_vec.csv', index=None)


