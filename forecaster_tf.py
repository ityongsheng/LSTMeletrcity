import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models_tf import create_lstm_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class EnergyForecasterTF:
    def __init__(self, input_shape, hidden_units=64, dropout_rate=0.2):
        """初始化TensorFlow能耗预测器"""
        self.model = create_lstm_model(input_shape, hidden_units, dropout_rate)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        使用早停和学习率调度器训练模型

        参数：
            X_train: 训练特征数据
            y_train: 训练标签数据
            X_val: 验证特征数据（可选）
            y_val: 验证标签数据（可选）
            epochs: 最大训练轮数
            batch_size: 批次大小
        """
        # 设置回调函数：早停和学习率调整
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
        ]

        # 准备验证数据（如果提供）
        validation_data = (X_val, y_val) if X_val is not None else None

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, X_test):
        """对测试数据进行预测"""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        计算模型的RMSE和MAPE评估指标

        参数：
            X_test: 测试特征数据
            y_test: 测试标签数据

        返回：
            包含RMSE和MAPE的字典
        """
        # 进行预测
        preds = self.predict(X_test)

        # 计算RMSE和MAPE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds)

        return {
            'RMSE': rmse,
            'MAPE': mape
        }