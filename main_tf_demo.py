import numpy as np
import pandas as pd
import tensorflow as tf
from power_processor import PowerDataProcessor
from forecaster_tf import EnergyForecasterTF
from explainer_tf import ExplainerTF
from test_processor import create_dummy_data
import os

def run_tf_demo():
    print("=== SHEMS TensorFlow/Keras Demo ===\n")
    
    # 1. Prepare Data using existing processor
    print("STEP 1: Data Preparation")
    demo_file = 'uk_dale_tf_demo.csv'
    create_dummy_data(demo_file, n_rows=5000)
    
    processor = PowerDataProcessor(demo_file)
    df = processor.load_and_clean_data(resample_freq='1H')
    df_features = processor.feature_engineering(lag_steps=[1, 2, 24], window_sizes=['6H', '24H'])
    processor.scale_data()
    
    T = 24
    H = 1
    X, y = processor.create_sequences(T=T, H=H)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Validation split from training
    val_split = int(0.8 * len(X_train))
    X_train_final, X_val = X_train[:val_split], X_train[val_split:]
    y_train_final, y_val = y_train[:val_split], y_train[val_split:]
    
    print(f"Train: {X_train_final.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 2. Model & Training
    print("\nSTEP 2: Training Keras LSTM")
    input_shape = (T, X.shape[2])
    forecaster = EnergyForecasterTF(input_shape=input_shape)
    
    history = forecaster.train(X_train_final, y_train_final, X_val, y_val, epochs=5, batch_size=16)
    
    # Save Model Weights for Service Layer
    forecaster.model.save_weights('shems_lstm.h5')
    print("Model weights saved to shems_lstm.h5")
    
    # 3. Evaluation
    print("\nSTEP 3: Evaluation")
    metrics = forecaster.evaluate(X_test, y_test)
    print(f"Test Metrics: RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.4f}")
    
    # 4. Explainability (SHAP)
    print("\nSTEP 4: SHAP Explainability")
    
    # TF2 / Eager execution is standard
    # tf.compat.v1.disable_v2_behavior() # Removed to allow proper Keras 3 / standard TF2 behavior

    # Background: Use a subset of training data
    background = X_train_final[:50]
    
    # Re-instantiate explainer after disable_v2_behavior might be needed depending on graph content, 
    # but disable_v2_behavior must be called BEFORE model creation usually. 
    # Since model is already created, switching to GradientExplainer is safer OR 
    # we just try to use ExplainerTF which internally tries DeepExplainer.
    # The error 'gradient registry has no entry' implies we need to run in legacy graph mode or use GradientExplainer.
    # Let's try forcing GradientExplainer inside explainer_tf.py instead, 
    # OR we just disable v2 behavior at START of script (but that breaks other things).
    
    # BETTER FIX: Use GradientExplainer which works better with TF2 eager.
    # Let's modify explainer_tf.py to prefer GradientExplainer if Deep fails, 
    # or just use GradientExplainer explicitly.
    
    explainer = ExplainerTF(forecaster.model, background)
    
    # Explain one sample from test set
    sample_idx = 0
    sample = X_test[sample_idx:sample_idx+1] # Keep dims (1, T, F)
    
    print("Computing SHAP values...")
    try:
        shap_values = explainer.compute_shap_values(sample)
        
        # Export for Frontend
        feature_names = [processor.target_col] + processor.feature_cols
        
        # Get base value using helper which handles different explainer types
        base_value = explainer.get_base_value()
            
        explainer.export_to_json(shap_values, base_value, sample, feature_names, output_path='shap_result.json')
        
        # Generate Summary Plot
        # We use a larger batch for summary plot
        summary_batch = X_test[:10]
        summary_shap = explainer.compute_shap_values(summary_batch)
        explainer.plot_summary(summary_shap, summary_batch, feature_names=feature_names, save_path='shap_summary_tf.png')
        print("SHAP Summary plot saved to shap_summary_tf.png")
        
    except Exception as e:
        print(f"SHAP execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tf_demo()
