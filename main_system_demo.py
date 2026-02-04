import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from power_processor import PowerDataProcessor
from forecaster import EnergyForecaster
from detector import AnomalyDetector
from explainer import XAIEngine
import torch

def run_shems_demo():
    print("=== SHEMS (Smart Home Energy Management System) Demo ===\n")
    
    # 1. Prepare Data
    print("STEP 1: Data Preparation")
    # Increase dummy data to ensure we have enough points after resampling 1H
    from test_processor import create_dummy_data
    create_dummy_data('uk_dale_demo.csv', n_rows=10000) 
    
    processor = PowerDataProcessor('uk_dale_demo.csv')
    df = processor.load_and_clean_data(resample_freq='1H')
    df_features = processor.feature_engineering(lag_steps=[1, 2, 24], window_sizes=['6H', '24H'])
    processor.scale_data()
    
    # Create sequences: History=24h, Horizon=1
    T = 24
    H = 1
    X, y = processor.create_sequences(T=24, H=1)
    
    # Split Train/Test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}\n")
    
    if len(X_train) < 5 or len(X_test) < 5:
        print("Not enough data generated for training/testing. Exiting.")
        return

    # 2. Forecasting Model
    print("STEP 2: Training LSTM Forecaster (Simulated)")
    input_dim = X.shape[2]
    forecaster = EnergyForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1, dropout=0.2)
    
    # Short training for demo
    loss_hist = forecaster.train(X_train, y_train, epochs=5, batch_size=16)
    
    # Predict with Uncertainty on Test Set
    print("Predicting with Uncertainty (MC Dropout)...")
    mean_preds, lower_ci, upper_ci = forecaster.predict(X_test, n_samples=20)
    
    # 3. Anomaly Detection
    print("\nSTEP 3: Anomaly Detection")
    # Set thresholds based on scaled data [0, 1] approximately
    detector = AnomalyDetector(static_threshold=0.95, sigma_factor=2.0)
    
    anomalies = []
    # Simulate real-time loop over test set
    for t in range(len(y_test)):
        actual_val = y_test[t].item()  # Extract scalar
        pred_mean = mean_preds[t].item()
        pred_std = (upper_ci[t].item() - lower_ci[t].item()) / 4 # Approx std from CI width
        
        # Inject synthetic anomaly at t=20
        if t == 20:
            actual_val = 1.5 # Spike
            print(f"Injecting anomaly at step {t}...")
            
        result = detector.detect(t, actual_val, pred_mean, pred_std)
        if result['is_anomaly']:
            anomalies.append(result)
            if len(anomalies) <= 3: # Print first few
                print(f"Anomaly at {t}: {result['alerts']}")

    print(f"Total anomalies detected: {len(anomalies)}")
    
    # 4. Explainability (SHAP)
    print("\nSTEP 4: Explainability (SHAP)")
    # Use training data as background
    background = X_train[:50] 
    final_feature_names = [processor.target_col] + processor.feature_cols
    
    xai = XAIEngine(forecaster.model, background, feature_names=final_feature_names)
    
    # Explain the anomaly point (or nearby)
    anomaly_idx = 20 if len(X_test) > 20 else 0
    sample = X_test[anomaly_idx]
    
    print("Generating Local Explanation...")
    try:
        shap_values = xai.explain_local(sample)
        explanation_text = xai.explain_in_text(shap_values, sample_idx=0)
        print(f"Explanation for Step {anomaly_idx}:")
        print(explanation_text)
    except Exception as e:
        print(f"Skipping XAI due to error: {e}")
    
    # 5. Visualization (Dashboard)
    print("\nSTEP 5: Dashboard Visualization")
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Forecast & CI
    plt.subplot(2, 1, 1)
    
    # Safe subset logic
    subset = min(100, len(y_test))
    
    plt.plot(range(subset), y_test[:subset], label='Actual', color='black')
    plt.plot(range(subset), mean_preds[:subset], label='Predicted', color='blue')
    plt.fill_between(range(subset), lower_ci[:subset].flatten(), upper_ci[:subset].flatten(), color='blue', alpha=0.2, label='95% CI')
    
    # Highlight anomalies
    anomaly_indices = [res['timestamp'] for res in anomalies if res['timestamp'] < subset]
    if anomaly_indices:
         # Need to align anomaly values. Since we can't easily access the injected values from loop
         # unless we stored them, we'll just plot points at proper x with actual y_test values, 
         # but note y_test does NOT contain the injected anomaly value 1.5. 
         # So we plot a mark at the detected time height = observed (which was passed to detect).
         # We stored 'power' in anomalies list.
         anomaly_vals = [res['power'] for res in anomalies if res['timestamp'] < subset]
         plt.scatter(anomaly_indices, anomaly_vals, color='red', marker='x', s=100, label='Anomaly')

         
    plt.title('SHEMS Real-time Forecast & Anomaly Detection')
    plt.legend()
    
    # Subplot 2: Local Feature Importance (SHAP Bar)
    plt.subplot(2, 1, 2)
    # Plot simple bar chart of the SHAP values for the anomaly sample
    if 'shap_values' in locals():
        # Average over time steps for the single sample -> (Features,)
        # shap_values: (1, T, F) or (T, F)
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        if sv.ndim == 3: sv = sv[0] # (T, F)
        
        # Mean absolute impact per feature
        imp = np.abs(sv).mean(axis=0)
        sorted_idx = np.argsort(imp)
        
        y_pos = np.arange(len(sorted_idx))
        plt.barh(y_pos, imp[sorted_idx])
        plt.yticks(y_pos, [final_feature_names[i] for i in sorted_idx])
        plt.title('Feature Contributions (SHAP) for Selected Sample')
        
    plt.tight_layout()
    plt.savefig('shems_dashboard.png')
    print("Dashboard saved to shems_dashboard.png")
    
if __name__ == "__main__":
    run_shems_demo()
