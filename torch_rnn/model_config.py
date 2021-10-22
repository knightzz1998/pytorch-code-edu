# -*- coding: utf-8 -*-
# 作者 : 王天赐
import torch


class RNNConfig():
    def __init__(self) -> None:
        self.vocab_dict_size = 1000  # 字典的大小, 字典就是对数据分词去重后的每个不重复出现的单词
        self.embed_size = 100  # 每一个step对应的tensor的的长度
        self.hidden_size = 128  # 隐藏层节点的数目
        self.num_layers = 3  # lstm 层的数量
        self.pad_size = 67  # 句子的最大长度, 不满足这个长度将会进行padding, 也就是填充<pad>
        self.num_classes = 2
        self.epochs = 30
        self.learn_rate = 0.01
        self.devices = torch.device('cuda')


class Word2VecConfig():
    def __init__(self) -> None:
        self.vector_size = 100  # 生成词向量的大小
        self.window = 6  # 窗口大小
        self.workers = 5  # 线程数
        self.min_count = 1  # 最小词频数
