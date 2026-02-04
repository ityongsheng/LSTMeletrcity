import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models import LSTMModel
from typing import Tuple, Optional


class EnergyForecaster:
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2, learning_rate=0.001):
        # 设置设备：优先使用GPU，否则使用CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化LSTM模型并移动到相应设备
        self.model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout).to(self.device)
        self.criterion = nn.MSELoss()  # 使用均方误差作为损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # Adam优化器

    def train(self, X_train, y_train, epochs=10, batch_size=32, verbose=True):
        """
        训练LSTM模型

        参数：
            X_train: 训练特征数据，Numpy数组，形状为(样本数, 时间步长, 特征数)
            y_train: 训练标签数据，Numpy数组，形状为(样本数, 预测步长)
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练进度信息
        """
        self.model.train()  # 将模型设置为训练模式

        # 将数据转换为PyTorch张量并移动到相应设备
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        # 创建TensorDataset和DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_history = []  # 存储损失历史

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()  # 清零梯度
                outputs = self.model(batch_X)  # 前向传播
                loss = self.criterion(outputs, batch_y)  # 计算损失
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数
                epoch_loss += loss.item()

            # 计算平均损失
            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)

            # 每5轮打印一次训练信息
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        return loss_history  # 返回损失历史

    def predict(self, X_test, n_samples=50):
        """
        进行预测并量化不确定性(使用MC Dropout)

        返回：
            mean: 预测均值，形状为(batch, output_dim)
            lower_ci: 95%置信区间下限，形状为(batch, output_dim) - 2个标准差
            upper_ci: 95%置信区间上限，形状为(batch, output_dim) + 2个标准差
        """
        # 重要：保持dropout激活以进行不确定性量化
        self.model.train()

        # 将测试数据转换为张量
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        # 使用模型的MC Dropout方法进行预测
        mean_pred, std_pred, _ = self.model.predict_with_uncertainty(X_tensor, n_samples=n_samples)

        # 将结果转换为numpy数组
        mean_pred = mean_pred.cpu().numpy()
        std_pred = std_pred.cpu().numpy()

        # 计算95%置信区间(近似2个标准差)
        lower_ci = mean_pred - 2 * std_pred
        upper_ci = mean_pred + 2 * std_pred

        return mean_pred, lower_ci, upper_ci

    def save_model(self, path):
        """保存模型参数到指定路径"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")

    def load_model(self, path):
        """从指定路径加载模型参数"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # 将模型设置为评估模式
        print(f"模型已从 {path} 加载")