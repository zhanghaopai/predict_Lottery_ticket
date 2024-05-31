import argparse

import torch
import torch.nn as nn

from dataloader.SSQDataLoader import build_dataloader
from model.LSTM import LSTMModel

learning_rate = 0.01
epochs = 50

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.path = 'data/ssq/data.csv'
args.train_ratio = 0.8
args.batch_size=100
train_loader, test_loader = build_dataloader(args)


def train():
    real_model = LSTMModel(input_dim=7, hidden_dim=5, num_layers=5, output_dim=1)
    real_optimizer = torch.optim.Adam(real_model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        real_model.train()
        epoch_train_loss = 0
        for i, info in enumerate(train_loader):
            # 梯度清零
            real_optimizer.zero_grad()
            # 前向传播
            output = real_model(info)
            # 计算loss，优化目标
            batch_loss = criterion(output, )

            # 反向传播与优化
            batch_loss.backward()
            real_optimizer.step()
            # 累加损失
            epoch_train_loss += batch_loss.item()
        print(f'Epoch [{epoch + 1}/{epoch}], Loss: {epoch_train_loss:.4f}')


if __name__=="__main__":
    train()
