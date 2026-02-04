import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input


def create_lstm_model(input_shape, hidden_units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    使用Functional API构建多层LSTM模型，包含批量归一化和Dropout层

    参数：
        input_shape: 元组 (时间步长, 特征数)
        hidden_units: 整数，LSTM层中的单元数量
        dropout_rate: 浮点数，Dropout比例
        learning_rate: 浮点数，学习率

    返回：
        编译好的Keras模型
    """
    # 输入层
    inputs = Input(shape=input_shape)

    # 第一层：LSTM + 批量归一化 + Dropout
    x = LSTM(hidden_units, return_sequences=True, activation='tanh')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # 第二层：LSTM + Dropout
    x = LSTM(hidden_units, return_sequences=False, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)

    # 输出层
    outputs = Dense(1)(x)

    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)

    # 使用Adam优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model