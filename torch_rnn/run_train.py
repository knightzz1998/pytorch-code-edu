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

    # 创建 loss 对象
    loss_func = nn.CrossEntropyLoss()

    # 创建优化函数
    optimizer = optim.Adam(rnn_model.parameters(), lr=config.learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=0.9)

    for epoch in range(config.epochs):
        for i, batch in enumerate(tqdm(train_data)):
            label, data = batch
            # 将label转换为tensor, 并放到GPU中
            data = torch.tensor(data).to(config.devices)
            label = torch.tensor(label, dtype=torch.int64).to(config.devices)
            # 初始化梯度
            optimizer.zero_grad()
            # 获取预测值
            pred = rnn_model.forward(data)
            # 计算loss
            loss_val = loss_func(pred, label)
            # 打印进度
            # print("\nepoch is {}, val is {}".format(epoch, loss_val))
            # 反向传播
            loss_val.backward()
            # 更新参数
            optimizer.step()
        scheduler.step()
        print("\nepoch is {}, val is {}".format(epoch, loss_val))
        if epoch % 2 == 0:
            torch.save(rnn_model.state_dict(), "../out_dir/model_rnn/{}.pth".format(epoch))


if __name__ == '__main__':
    run()
