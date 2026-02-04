import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union

class PowerDataProcessor:
    """
    Processor for UK-DALE dataset handling data cleaning, feature engineering,
    and sequence preparation for LSTM models.
    """
    
    def __init__(self, file_path: str, datetime_col: str = 'timestamp'):
        """
        Initialize the processor.
        
        Args:
            file_path: Path to the CSV dataset.
            datetime_col: Name of the datetime column.
        """
        self.file_path = file_path
        self.datetime_col = datetime_col
        self.df = None
        self.scaler = MinMaxScaler()
        self.feature_cols = []
        self.target_col = None
        
    def load_and_clean_data(self, target_col: str = 'aggregate', resample_freq: str = '1T') -> pd.DataFrame:
        """
        Load data, parse dates, set index, and handle missing values.
        
        Args:
            target_col: Name of the target power column.
            resample_freq: Frequency to resample data (e.g., '1T' for 1 minute).
            
        Returns:
            Cleaned DataFrame.
        """
        self.target_col = target_col
        print(f"Loading data from {self.file_path}...")
        
        # Optimize loading by parsing dates immediately
        try:
            self.df = pd.read_csv(self.file_path, parse_dates=[self.datetime_col])
        except ValueError:
             # Fallback if date parsing fails in read_csv
            self.df = pd.read_csv(self.file_path)
            self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
            
        self.df.set_index(self.datetime_col, inplace=True)
        
        # Resample to handle missing timestamps and ensure regular frequency
        # Use mean for downsampling, interpolate for upsampling (small gaps)
        original_shape = self.df.shape
        self.df = self.df.resample(resample_freq).mean()
        
        # Handle missing values: Interpolate for short gaps, forward fill for others
        self.df.interpolate(method='time', limit_direction='both', inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True) # Catch remaining
        
        print(f"Data loaded and cleaned. Shape changed from {original_shape} to {self.df.shape}")
        return self.df

    def feature_engineering(self, 
                            lag_steps: List[int] = [1, 2, 3], 
                            window_sizes: List[str] = ['1H', '24H'],
                            use_time_encoding: bool = True) -> pd.DataFrame:
        """
        Generate Lagged features, Rolling statistics, and Time encodings.
        
        Args:
            lag_steps: List of lag steps to create.
            window_sizes: List of time windows for rolling stats (e.g., '24H').
            use_time_encoding: Whether to add sin/cos time features.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_clean_data() first.")
            
        print("Starting feature engineering...")
        
        # 1. External Features (Simulated check, normally user ensures these exist or passes them)
        # For this class we assume they might exist in source or we skip.
        
        # 2. Lag Features
        for lag in lag_steps:
            self.df[f'lag_{lag}'] = self.df[self.target_col].shift(lag)
            
        # 3. Rolling Statistics
        for window in window_sizes:
            # Note: Rolling on time index requires strict frequency or valid index
            # min_periods=1 ensures we get values even at start
            roller = self.df[self.target_col].rolling(window=window, min_periods=1)
            self.df[f'rolling_mean_{window}'] = roller.mean()
            self.df[f'rolling_std_{window}'] = roller.std()
            
        # 4. Time Encoding (Cyclical features)
        if use_time_encoding:
            # Hour of day
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df.index.hour / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df.index.hour / 24)
            # Day of week
            self.df['day_sin'] = np.sin(2 * np.pi * self.df.index.dayofweek / 7)
            self.df['day_cos'] = np.cos(2 * np.pi * self.df.index.dayofweek / 7)
            # Month
            self.df['month_sin'] = np.sin(2 * np.pi * self.df.index.month / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df.index.month / 12)

        # Drop NaNs created by lagging
        # For very large lags, we might lose early data.
        self.df.dropna(inplace=True)
        
        # Update feature columns list (excluding target for now)
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]
        
        print(f"Feature engineering complete. Total features: {len(self.feature_cols)}")
        return self.df

    def scale_data(self) -> pd.DataFrame:
        """
        Normalize the dataframe features and target.
        """
        print("Scaling data...")
        # Fit on all data for simplicity (In production, fit only on train!)
        # We assume the user splits before or we provide a split method.
        # Here we scale the whole DF for formatting purposes.
        self.df_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df),
            columns=self.df.columns,
            index=self.df.index
        )
        return self.df_scaled

    def fit_scaler(self, train_df: pd.DataFrame):
        """Fit scaler only on training data to avoid leakage."""
        self.scaler.fit(train_df)
        
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        return pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns,
            index=df.index
        )

    def create_sequences(self, 
                         T: int, 
                         H: int, 
                         feature_cols: Optional[List[str]] = None,
                         data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (Samples, Time_steps, Features) sequences using efficient striding.
        
        Args:
            T: Input sequence length (History).
            H: Output prediction steps (Horizon).
            feature_cols: List of feature columns to use. Defaults to all generated features.
            data: Optional DataFrame to use. If None, uses self.df_scaled (or self.df).
        
        Returns:
            X: Input sequences of shape (Samples, T, Features)
            y: Targets of shape (Samples, H)
        """
        if data is None:
            data = self.df_scaled if hasattr(self, 'df_scaled') else self.df
        
        if feature_cols is None:
             # Use all columns including target as features? Usually yes for autoregression
             # Or user specific specific exogenous features. 
             # Let's default to using ALL columns currently in dataframe as features.
            feature_subset = data.values
        else:
            feature_subset = data[feature_cols].values
            
        target_subset = data[[self.target_col]].values
        
        # Efficient sequence generation using sliding_window_view (Available in numpy 1.20+)
        # Or manual stride tricks for backward compatibility if needed, but 1.20 is standard now.
        
        # Let's implementation a robust loop-free method
        # X: [t-T, ..., t-1]
        # y: [t, ..., t+H-1]
        
        # Total samples = len(data) - T - H + 1
        n_samples = len(data) - T - H + 1
        
        if n_samples <= 0:
            raise ValueError(f"Data length {len(data)} is too small for T={T}, H={H}")
            
        # Create indices
        # shape (n_samples, T)
        input_indices = np.arange(n_samples)[:, None] + np.arange(T)[None, :]
        # shape (n_samples, H)
        target_indices = np.arange(n_samples)[:, None] + np.arange(T, T + H)[None, :]
        
        X = feature_subset[input_indices]
        y = target_subset[target_indices]
        
        # Squeeze y if H=1 for convenience (Samples, ) instead of (Samples, 1)
        if H == 1:
            y = y.reshape(-1)
        else:
            y = y.reshape(n_samples, H)
            
        print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def visualize_eda(self, save_path: Optional[str] = None):
        """
        Basic EDA plots.
        """
        if self.df is None:
            print("No data to visualize.")
            return

        plt.figure(figsize=(15, 10))
        
        # 1. Power Time Series
        plt.subplot(2, 2, 1)
        plt.plot(self.df.index, self.df[self.target_col], label='Power')
        plt.title('Power Consumption Over Time')
        plt.legend()
        
        # 2. Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(self.df[self.target_col], kde=True)
        plt.title('Power Distribution')
        
        # 3. Correlation Matrix (Top 10 features)
        plt.subplot(2, 2, 3)
        corr = self.df.corr()
        # sort by correlation with target
        top_corr_cols = corr[self.target_col].abs().sort_values(ascending=False).head(10).index
        sns.heatmap(corr.loc[top_corr_cols, top_corr_cols], annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap')
        
        # 4. Rolling Mean vs Raw (First 1000 pts for visibility)
        plt.subplot(2, 2, 4)
        subset = self.df.head(1000)
        plt.plot(subset.index, subset[self.target_col], alpha=0.5, label='Raw')
        # Check if we have rolling features
        rolling_cols = [c for c in self.df.columns if 'rolling_mean' in c]
        if rolling_cols:
            plt.plot(subset.index, subset[rolling_cols[0]], color='red', label=rolling_cols[0])
        plt.title('Raw vs Rolling Mean (First 1000 steps)')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

# Example usage pattern included in __main__ for quick test if run directly
if __name__ == "__main__":
    pass
