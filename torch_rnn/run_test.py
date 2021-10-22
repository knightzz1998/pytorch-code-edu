# -*- coding: utf-8 -*-
# 作者 : 王天赐

# -*- coding: utf-8 -*-
# 作者 : 王天赐
import torch
from torch import nn
from torch import optim
from model_classification import RNNModel
from tqdm import tqdm


def run():
    from model_config import RNNConfig
    from datasets import load_dataset, data_loader

    config = RNNConfig()

    # 获取数据集
    dataset = load_dataset()
    train_data = data_loader(dataset)

    # 设置pad_size
    config.pad_size = dataset.max_seq_len

    # 创建模型对象
    rnn_model = RNNModel(config, dataset.weights)
    # 将模型数据放到GPU上
    rnn_model.to(config.devices)

    # 加载模型
    rnn_model.load_state_dict(torch.load("../out_dir/model_rnn/28.pth"))


    # 创建优化函数
    for i, batch in enumerate(train_data):
        label, data = batch
        # 将label转换为tensor, 并放到GPU中
        data = torch.tensor(data).to(config.devices)
        label = torch.tensor(label, dtype=torch.int64).to(config.devices)
        # 获取预测值
        pred = rnn_model.forward(data)

        pred  = torch.argmax(pred, dim=1)
        out = torch.eq(pred, label)
        accuracy = out.sum() * 1.0  / pred.size()[0]
        print("\n accuracy : {}".format(accuracy))


if __name__ == '__main__':
    run()
