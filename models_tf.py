import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input

def create_lstm_model(input_shape, hidden_units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Builds a Multi-layer LSTM model with BatchNormalization and Dropout using Functional API.
    
    Args:
        input_shape: tuple (time_steps, features)
        hidden_units: int, number of units in LSTM layers
        dropout_rate: float, dropout fraction
        learning_rate: float
        
    Returns:
        Compiled Keras Model
    """
    inputs = Input(shape=input_shape)
    
    # Layer 1: LSTM + BatchNorm + Dropout
    x = LSTM(hidden_units, return_sequences=True, activation='tanh')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Layer 2: LSTM + Dropout
    x = LSTM(hidden_units, return_sequences=False, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output Layer
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model
