# -*- coding: utf-8 -*-
# 作者 : 王天赐
import torch
from torch import nn


class RNNModel(nn.Module):

    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        """
            定义模型
        :param vocab:
        :param embed_size:
        :param num_hiddens:
        :param num_layers:
        """
        super(RNNModel).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        """
            使用模型
        :param input:
        :return:
        """
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs
