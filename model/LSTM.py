import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出用于分类
        out = self.fc(out[:, -1, :])
        return out


if __name__=="__main__":
    # 定义模型参数
    input_dim = 10  # 输入特征维度
    hidden_dim = 50  # LSTM隐藏层维度
    num_layers = 2  # LSTM层数
    output_dim = 1  # 输出维度

    # 创建模型实例
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

    # 创建一个随机数据样本来测试模型
    x = torch.rand(5, 3, input_dim)  # 假设batch_size=5, 序列长度=3
    output = model(x)
    print(output)