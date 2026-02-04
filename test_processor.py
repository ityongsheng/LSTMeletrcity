import pandas as pd
import numpy as np
import os
from power_processor import PowerDataProcessor

def create_dummy_data(filename='dummy_uk_dale.csv', n_rows=2000):
    """Create a synthetic dataset for testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1T')
    
    # Simulate data
    # Aggregate power: mostly noise + daily pattern
    t = np.arange(n_rows)
    daily_pattern = 500 + 200 * np.sin(2 * np.pi * t / (24*60))
    noise = np.random.normal(0, 50, n_rows)
    power = daily_pattern + noise
    # Add some anomalies/spikes
    power[::100] += 500
    
    # Temperature: slower moving
    temp = 15 + 5 * np.sin(2 * np.pi * t / (24*60*3)) + np.random.normal(0, 1, n_rows)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'aggregate': power,
        'temperature': temp,
        'holiday': [0] * n_rows # simple constant for test
    })
    
    # Introduce some missing values to test cleaning
    df.loc[100:105, 'aggregate'] = np.nan
    
    df.to_csv(filename, index=False)
    print(f"Created dummy data: {filename} with shape {df.shape}")
    return filename

def test_processor():
    dummy_file = 'dummy_uk_dale.csv'
    create_dummy_data(dummy_file)
    
    try:
        # 1. Initialize
        processor = PowerDataProcessor(file_path=dummy_file)
        
        # 2. Load & Clean
        print("\n--- Testing Load & Clean ---")
        processor.load_and_clean_data(resample_freq='1H') # Downsample to 1H for speed/test
        print("Data loaded. Head:")
        print(processor.df.head())
        
        # Check for NaNs
        if processor.df.isna().sum().sum() == 0:
            print("PASS: No missing values after cleaning.")
        else:
            print("FAIL: Found missing values.")
            
        # 3. Feature Engineering
        print("\n--- Testing Feature Engineering ---")
        processor.feature_engineering(lag_steps=[1, 2], window_sizes=['3H'], use_time_encoding=True)
        print("Columns after FE:", processor.df.columns.tolist())
        
        required_cols = ['lag_1', 'rolling_mean_3H', 'hour_sin']
        if all(col in processor.df.columns for col in required_cols):
            print("PASS: Required features generated.")
        else:
            print("FAIL: Missing features.")
            
        # 4. Scaling
        print("\n--- Testing Scaling ---")
        df_scaled = processor.scale_data()
        if (df_scaled.min().min() >= 0) and (df_scaled.max().max() <= 1.000001): # approximately [0, 1]
             print("PASS: Data scaled approx [0, 1].")
        else:
             print(f"WARN: Data range {df_scaled.min().min()} to {df_scaled.max().max()}")
        
        # 5. Sequences
        print("\n--- Testing Sequence Generation ---")
        # T=6 (6 hours history), H=1 (predict next hour)
        X, y = processor.create_sequences(T=6, H=1)
        
        expected_features = processor.df.shape[1]
        print(f"X shape: {X.shape}, expected last dim: {expected_features}")
        
        if X.shape[1] == 6 and X.shape[2] == expected_features:
            print("PASS: Sequence shape correct.")
        else:
            print("FAIL: Sequence shape incorrect.")
            
        # 6. EDA
        print("\n--- Testing EDA (Generation only) ---")
        processor.visualize_eda(save_path='test_eda.png')
        if os.path.exists('test_eda.png'):
            print("PASS: EDA plot saved.")
        else:
            print("FAIL: EDA plot not saved.")
            
    finally:
        # Cleanup
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        if os.path.exists('test_eda.png'):
            os.remove('test_eda.png')
        print("\nTest Complete.")

if __name__ == "__main__":
    test_processor()
