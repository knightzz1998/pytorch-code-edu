# -*- coding: utf-8 -*-
# 作者 : 王天赐
import jieba
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from tqdm import tqdm
from word2vec_model import Word2vecModel


class WeiboDataSets(Dataset):

    def __init__(self, data_path, stop_words_path) -> None:
        self.data_path = data_path
        self.stop_words_path = stop_words_path
        self.word2vec_model = Word2vecModel()

        self.weights = self.word2vec_model.get_vectors_array()  # 获取向量矩阵, 维度为100维
        self.word_dict = self.word2vec_model.get_word_dict()  # {单词:索引}
        self.word_list = self.word2vec_model.get_word_list()  # [单词, 单词, ...]

        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        # 添加PAD和UNK
        self.word_dict.update({self.UNK: len(self.word_dict), self.PAD: len(self.word_dict) + 1})
        # 添加到word_list中
        self.word_list.append(self.UNK)
        self.word_list.append(self.PAD)

        # 添加100维的0向量
        self.weights = torch.FloatTensor(np.vstack((self.weights, np.zeros(100), np.zeros(100))))

        # 读取停用词
        self.stop_words = self.read_stop_words()
        # 读取分词并过滤停用词后的数据
        self.data, self.max_seq_len = self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
            接收一个索引,返回一个样本
        :param index:
        :return:
        """
        item = self.data[index]
        label = item[0]
        word_list = item[1]
        input_idx = []

        # 将单词转换为向量表(词表)的索引, 词表和对应的向量表是对应的
        for word in word_list:
            if word in self.word_list:
                input_idx.append(self.word_dict[word])
            else:
                input_idx.append(self.word_dict[self.UNK])
        # 对数据进行padding, 填充长度不足的数据
        if len(input_idx) < self.max_seq_len:
            input_idx += [self.word_dict[self.PAD] for _ in range(self.max_seq_len - len(input_idx))]
        data = np.array(input_idx)
        return label, data

    def read_stop_words(self):
        """
            读取停用词
        :param self:
        :param stop_words_path:
        :return:
        """
        stop_words = open(self.stop_words_path, "r", encoding="utf-8")
        stop_words = [stop_word.strip() for stop_word in stop_words]
        stop_words.append(" ")
        stop_words.append("\n")
        return stop_words

    def read_data(self):
        # 读取评论数据
        data_list = pd.read_csv(self.data_path).values.tolist()  # 将DataFrame转换为List
        # 读取停用词
        stop_words = self.stop_words
        # 对数据进行shuffle
        np.random.shuffle(data_list)

        data = []  # 存储分词并过滤停用词后的数据
        max_seq_len = 0  # 统计句子最长的长度

        for item in tqdm(data_list):
            sentence = item[1]
            label = item[0]
            seq_list = jieba.cut(sentence.strip(), cut_all=False)
            seq_res = []
            # 去除停用词
            for seq in seq_list:
                if seq in stop_words:
                    continue
                seq_res.append(seq)
            if len(seq_res) > max_seq_len:
                max_seq_len = len(seq_res)
            data.append([label, seq_res])
        return data, max_seq_len


def data_loader(dataset):
    return DataLoader(dataset, batch_size=1024, shuffle=True)


def load_dataset():
    data_path = "../data/weibo_senti_100k.csv"
    stop_words_path = "../data/cn_stopwords.txt"
    dataset = WeiboDataSets(data_path, stop_words_path)
    return dataset


if __name__ == '__main__':
    datasets = load_dataset()
    train_dataloader = data_loader(datasets)
    for i, batch in enumerate(train_dataloader):
        print(batch[1].size())
