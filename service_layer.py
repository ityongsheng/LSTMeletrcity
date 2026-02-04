import tensorflow as tf
from models_tf import create_lstm_model
from explainer_tf import ExplainerTF
from detector import AnomalyDetector
from power_processor import PowerDataProcessor
import numpy as np
import os
import joblib # or pickle used if scaler persistence is needed

class ShemsService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ShemsService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        print("Initializing SHEMS Service Layer...")
        # 1. Load Processor (Stateful: Scaler)
        # Ideally we load a saved scaler. For demo, we refit or load dummy.
        # We'll re-use the dummy generator logic to initialize scaler for now.
        # In prod: self.processor = PowerDataProcessor.load('saved_processor.pkl')
        self.processor = PowerDataProcessor('uk_dale_tf_demo.csv') 
        # Hack: We assume data exists from main_tf_demo.py run
        if os.path.exists('uk_dale_tf_demo.csv'):
            self.processor.load_and_clean_data()
            self.processor.feature_engineering(lag_steps=[1, 2, 24], window_sizes=['6H', '24H'])
            self.processor.scale_data()
        else:
            print("WARN: Demo data not found. Scaler uninitialized.")
            
        # 2. Load Model
        # Input shape needs to match training T=24, Features=...
        if getattr(self.processor, 'df_scaled', None) is not None:
             input_dim = len(self.processor.feature_cols) + 1 # +1 for target if included
             # Actually create_sequences uses all cols if not specified
             input_dim = self.processor.df_scaled.shape[1] 
        else:
             input_dim = 16 # Fallback based on demo
             
        self.T = 24
        self.model = create_lstm_model((self.T, input_dim))
        
        # Load weights if exist
        if os.path.exists('shems_lstm.h5'): # We need to ensure we saved it in main_tf_demo
            self.model.load_weights('shems_lstm.h5')
            print("Model weights loaded.")
        else:
            print("WARN: No model weights found. Using random init.")

        # 3. Load Explainer
        # Explainer needs background data. We take a snippet from processor data.
        if hasattr(self.processor, 'df_scaled'):
            # Convert to sequences
            X, _ = self.processor.create_sequences(T=self.T, H=1)
            background = X[:50]
            self.explainer = ExplainerTF(self.model, background)
        else:
            self.explainer = None
            
        # 4. Detector
        self.detector = AnomalyDetector(static_threshold=0.9, sigma_factor=3.0) # threshold scaled approx
        
        self.initialized = True
        print("SHEMS Service Layer Ready.")

    def get_latest_data(self):
        """Get the latest sequence from loaded data for demo purposes."""
        if not hasattr(self.processor, 'df_scaled'): return None
        X, y = self.processor.create_sequences(T=self.T, H=1)
        # Return last sample
        return X[-1:], y[-1]

    def predict(self, input_seq):
        """
        Args:
            input_seq: (1, T, F) numpy array
        Returns:
            dict: {mean, lower, upper}
        """
        # MC Dropout Inference manually or via forecaster util
        # We can implement simple loop here
        preds = []
        for _ in range(20):
            preds.append(self.model(input_seq, training=True).numpy())
        
        preds = np.array(preds) # (20, 1, 1)
        mean = preds.mean()
        std = preds.std()
        
        return {
            'forecast': float(mean),
            'ci_low': float(mean - 2*std),
            'ci_high': float(mean + 2*std),
            'std': float(std)
        }

    def explain(self, input_seq):
        """
        Args:
           input_seq: (1, T, F)
        Returns:
           dict: JSON-able SHAP data
        """
        if not self.explainer: return {}
        
        shap_values = self.explainer.compute_shap_values(input_seq)
        base_value = self.explainer.get_base_value()
        
        feature_names = [self.processor.target_col] + self.processor.feature_cols
        
        # Calculate aggregate contribution per feature
        # shap_values shape: (T, F) (since we passed 1 sample and got [0])
        # Force plot usually wants simple vector inputs (features, values)
        
        # We aggregate over time for simple bar chart
        sv_agg = shap_values.sum(axis=0) if hasattr(shap_values, 'sum') else shap_values
        feat_agg = input_seq[0].mean(axis=0)
        
        return {
            "base_value": float(base_value),
            "shap_values": sv_agg.tolist(),
            "features": feat_agg.tolist(),
            "feature_names": feature_names
        }

    def alert(self, actual_value, predict_result):
        """
        Check anomaly.
        """
        return self.detector.detect(
            current_timestamp="now", 
            current_power=actual_value,
            predicted_mean=predict_result['forecast'],
            predicted_std=predict_result['std']
        )
