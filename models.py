import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim  # LSTM隐藏层维度
        self.num_layers = num_layers  # LSTM层数

        # LSTM层
        # batch_first=True -> 输入输出形状为(batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Dropout层（用于MC Dropout，在需要时显式应用到全连接部分，
        # 但nn.LSTM的dropout只影响层间连接，不影响最后输出）
        self.dropout = nn.Dropout(dropout)

        # 全连接层，将LSTM输出映射到目标维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 用零初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM前向传播
        # out形状: (batch, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))

        # 解码最后一个时间步的隐藏状态
        # out[:, -1, :] 形状: (batch, hidden_dim)
        last_step_out = out[:, -1, :]

        # 在推理阶段应用dropout以实现MC Dropout效果
        last_step_out = self.dropout(last_step_out)

        # 全连接层
        prediction = self.fc(last_step_out)
        return prediction

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        执行蒙特卡洛Dropout推理

        返回：
            mean_pred: 预测均值
            std_pred: 预测标准差
            predictions: 所有采样预测结果
        """
        self.train()  # 启用dropout
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(self.forward(x).unsqueeze(0))

        # 形状: (n_samples, batch, output_dim)
        predictions = torch.cat(predictions, dim=0)

        # 计算统计量
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred, predictions