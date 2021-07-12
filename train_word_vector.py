# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 22:48:46 2021

@author: yaoyanzhi
"""
# *****************************训练词向量*********************************
'''载入数据'''
import pandas as pd

train = pd.read_csv("labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)  # 载入有标记的训练数据

test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)  # 载入测试数据

unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0,
                              delimiter="\t", quoting=3)  # 载入未标记的训练数据

'''清洗数据'''
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


def review_to_wordlist(review, remove_stopwords=False):
    # 对原始评论进行清洗，并分割为单词

    review_text = BeautifulSoup(review).get_text()  # 删除HTML标记

    review_text = re.sub("[^a-zA-Z]", " ", review_text)  # 仅保留字母

    words = review_text.lower().split()  # 句子分割为单词并全部小写

    if remove_stopwords:  # 去停用词，默认关闭
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return (words)


'''分割段落为句子'''
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  # 载入句子分割器


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    '''
    Parameters
    ----------
    review : 原始评论
    tokenizer : 句子分割器
    remove_stopwords : 去停用词，默认关闭
    
    Returns
    -------
    sentences : 列表嵌套列表，内部每个列表是一个句子，且每个句子分割为单词

    '''
    raw_sentences = tokenizer.tokenize(review.strip())  # 将原始评论分割为若干句子

    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, \
                                                remove_stopwords))

    return sentences


sentences = []  # 初始化一个空的句子集合

for review in train["review"]:  # 加入标记训练集的评论
    sentences += review_to_sentences(review, tokenizer)

for review in unlabeled_train["review"]:  # 加入为标记训练集的评论
    sentences += review_to_sentences(review, tokenizer)

'''训练并保存自己的模型'''

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                    level=logging.INFO)

# 初始化和训练模型
from gensim.models import word2vec

model = word2vec.Word2Vec(sentences,
                          sg=0,  # 架构，0为分层CBOW，1为skip-gram
                          hs=0,  # 训练算法，0为分词softmax，1为负采样
                          workers=4,  # 线程数量
                          size=300,  # 词向量维数
                          min_count=40,  # 最低词频，以现直词汇数量
                          window=10,  # 窗口大小
                          sample=1e-3)  # 常用单词的下采样设置
# 保存训练好的模型
model_name = "300features_40minwords_10context"
model.save(model_name)
