import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models_tf import create_lstm_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

class EnergyForecasterTF:
    def __init__(self, input_shape, hidden_units=64, dropout_rate=0.2):
        self.model = create_lstm_model(input_shape, hidden_units, dropout_rate)
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train with Early Stopping and LR Scheduler.
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
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
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        """
        Calculate RMSE and MAPE.
        """
        preds = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds)
        
        return {
            'RMSE': rmse,
            'MAPE': mape
        }
