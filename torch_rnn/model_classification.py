# -*- coding: utf-8 -*-
# 作者 : 王天赐

import torch
from torch import nn
from torch.nn import Module
from torch.nn import LSTM, Linear, Softmax
from torch.nn import functional


class RNNModel(Module):

    def __init__(self, config, weights):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = LSTM(input_size=100,
                         hidden_size=128,
                         num_layers=3)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.linear = Linear(config.hidden_size, config.num_classes)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        embed = self.embedding(x)  # [数据单个批次大小, 句子长度, 单词向量长度]
        out, hidden = self.lstm(embed)
        out = functional.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out)
        out = out.reshape(out.size()[0], -1)
        out = self.linear(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    from model_config import RNNConfig
    from datasets import load_dataset, data_loader

    config = RNNConfig()

    dataset = load_dataset()
    train_data = data_loader(dataset)
    config.pad_size = dataset.max_seq_len

    rnn_model = RNNModel(config, dataset.weights)

    for batch in train_data:
        print(batch[1].size())
        pred = rnn_model.forward(batch[1])
        print(pred.size())
