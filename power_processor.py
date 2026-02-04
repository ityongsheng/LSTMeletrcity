import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union


class PowerDataProcessor:
    """
    UK-DALE数据集处理器，负责数据清洗、特征工程，
    以及为LSTM模型准备序列数据。
    """

    def __init__(self, file_path: str, datetime_col: str = 'timestamp'):
        """
        初始化处理器

        参数：
            file_path: CSV数据集路径
            datetime_col: 日期时间列的名称
        """
        self.file_path = file_path
        self.datetime_col = datetime_col
        self.df = None  # 原始数据
        self.scaler = MinMaxScaler()  # 归一化器
        self.feature_cols = []  # 特征列列表
        self.target_col = None  # 目标列名称

    def load_and_clean_data(self, target_col: str = 'aggregate', resample_freq: str = '1T') -> pd.DataFrame:
        """
        加载数据，解析日期，设置索引，并处理缺失值

        参数：
            target_col: 目标功率列的名称
            resample_freq: 数据重采样频率（例如：'1T'表示1分钟）

        返回：
            清洗后的DataFrame
        """
        self.target_col = target_col
        print(f"从 {self.file_path} 加载数据...")

        # 通过立即解析日期来优化加载
        try:
            self.df = pd.read_csv(self.file_path, parse_dates=[self.datetime_col])
        except ValueError:
            # 如果在read_csv中日期解析失败，回退方案
            self.df = pd.read_csv(self.file_path)
            self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])

        self.df.set_index(self.datetime_col, inplace=True)

        # 重采样以处理缺失时间戳并确保规则的频率
        # 使用均值进行降采样，使用插值进行上采样（小间隔）
        original_shape = self.df.shape
        self.df = self.df.resample(resample_freq).mean()

        # 处理缺失值：短间隔使用插值，其他使用前向填充
        self.df.interpolate(method='time', limit_direction='both', inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)  # 捕获剩余的NaN值

        print(f"数据已加载并清洗。形状从 {original_shape} 变为 {self.df.shape}")
        return self.df

    def feature_engineering(self,
                            lag_steps: List[int] = [1, 2, 3],
                            window_sizes: List[str] = ['1H', '24H'],
                            use_time_encoding: bool = True) -> pd.DataFrame:
        """
        生成滞后特征、滚动统计量和时间编码特征

        参数：
            lag_steps: 要创建的滞后步长列表
            window_sizes: 滚动统计的时间窗口列表（例如：'24H'）
            use_time_encoding: 是否添加sin/cos时间特征
        """
        if self.df is None:
            raise ValueError("数据未加载。请先调用 load_and_clean_data() 方法。")

        print("开始特征工程...")

        # 1. 外部特征（模拟检查，通常用户确保这些特征存在或传入）
        # 对于这个类，我们假设它们可能存在于源数据中，或者我们跳过。

        # 2. 滞后特征
        for lag in lag_steps:
            self.df[f'lag_{lag}'] = self.df[self.target_col].shift(lag)

        # 3. 滚动统计
        for window in window_sizes:
            # 注意：在时间索引上滚动需要严格的频率或有效索引
            # min_periods=1确保我们在开始时也能获得值
            roller = self.df[self.target_col].rolling(window=window, min_periods=1)
            self.df[f'rolling_mean_{window}'] = roller.mean()
            self.df[f'rolling_std_{window}'] = roller.std()

        # 4. 时间编码（周期性特征）
        if use_time_encoding:
            # 一天中的小时
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df.index.hour / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df.index.hour / 24)
            # 一周中的天
            self.df['day_sin'] = np.sin(2 * np.pi * self.df.index.dayofweek / 7)
            self.df['day_cos'] = np.cos(2 * np.pi * self.df.index.dayofweek / 7)
            # 月份
            self.df['month_sin'] = np.sin(2 * np.pi * self.df.index.month / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df.index.month / 12)

        # 删除由滞后操作创建的NaN值
        # 对于非常大的滞后步长，我们可能会丢失早期数据。
        self.df.dropna(inplace=True)

        # 更新特征列列表（暂时排除目标列）
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]

        print(f"特征工程完成。总特征数：{len(self.feature_cols)}")
        return self.df

    def scale_data(self) -> pd.DataFrame:
        """
        对DataFrame的特征和目标进行归一化
        """
        print("正在归一化数据...")
        # 为了简化，对整个数据集进行拟合（在生产中，只对训练集进行拟合！）
        # 我们假设用户在分割之前调用此方法，或者我们提供分割方法。
        # 这里为了格式化目的，我们对整个DataFrame进行归一化。
        self.df_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df),
            columns=self.df.columns,
            index=self.df.index
        )
        return self.df_scaled

    def fit_scaler(self, train_df: pd.DataFrame):
        """仅在训练数据上拟合归一化器，以避免数据泄漏"""
        self.scaler.fit(train_df)

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的归一化器转换数据"""
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
        使用高效的滑动窗口创建 (样本数, 时间步长, 特征数) 序列

        参数：
            T: 输入序列长度（历史长度）
            H: 输出预测步长（预测范围）
            feature_cols: 使用的特征列列表。默认为所有生成的特征。
            data: 要使用的DataFrame。如果为None，则使用self.df_scaled（或self.df）

        返回：
            X: 输入序列，形状为 (样本数, T, 特征数)
            y: 目标序列，形状为 (样本数, H)
        """
        if data is None:
            data = self.df_scaled if hasattr(self, 'df_scaled') else self.df

        if feature_cols is None:
            # 使用所有列（包括目标列）作为特征？对于自回归模型通常是这样的。
            # 或者用户指定特定的外生特征。
            # 这里默认使用当前DataFrame中的所有列作为特征。
            feature_subset = data.values
        else:
            feature_subset = data[feature_cols].values

        target_subset = data[[self.target_col]].values

        # 使用高效的滑动窗口方法生成序列（需要numpy 1.20+）
        # 如果需要向后兼容，可以使用手动步幅技巧，但1.20现在是标准。

        # 实现一种鲁棒的无循环方法
        # X: [t-T, ..., t-1]
        # y: [t, ..., t+H-1]

        # 总样本数 = len(data) - T - H + 1
        n_samples = len(data) - T - H + 1

        if n_samples <= 0:
            raise ValueError(f"数据长度 {len(data)} 对于 T={T}, H={H} 来说太小")

        # 创建索引
        # 形状 (n_samples, T)
        input_indices = np.arange(n_samples)[:, None] + np.arange(T)[None, :]
        # 形状 (n_samples, H)
        target_indices = np.arange(n_samples)[:, None] + np.arange(T, T + H)[None, :]

        X = feature_subset[input_indices]
        y = target_subset[target_indices]

        # 如果 H=1，为了方便，将 y 压缩为 (样本数,) 而不是 (样本数, 1)
        if H == 1:
            y = y.reshape(-1)
        else:
            y = y.reshape(n_samples, H)

        print(f"序列已创建。X 形状：{X.shape}，y 形状：{y.shape}")
        return X, y

    def visualize_eda(self, save_path: Optional[str] = None):
        """
        基本探索性数据分析图表
        """
        if self.df is None:
            print("没有可供可视化的数据。")
            return

        plt.figure(figsize=(15, 10))

        # 1. 功率时间序列图
        plt.subplot(2, 2, 1)
        plt.plot(self.df.index, self.df[self.target_col], label='功率')
        plt.title('随时间变化的功率消耗')
        plt.legend()

        # 2. 分布图
        plt.subplot(2, 2, 2)
        sns.histplot(self.df[self.target_col], kde=True)
        plt.title('功率分布')

        # 3. 相关性热力图（前10个特征）
        plt.subplot(2, 2, 3)
        corr = self.df.corr()
        # 按与目标列的相关性绝对值降序排序
        top_corr_cols = corr[self.target_col].abs().sort_values(ascending=False).head(10).index
        sns.heatmap(corr.loc[top_corr_cols, top_corr_cols], annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('特征相关性热力图')

        # 4. 滚动均值与原始值对比（前1000个点以便观察）
        plt.subplot(2, 2, 4)
        subset = self.df.head(1000)
        plt.plot(subset.index, subset[self.target_col], alpha=0.5, label='原始值')
        # 检查是否有滚动特征
        rolling_cols = [c for c in self.df.columns if 'rolling_mean' in c]
        if rolling_cols:
            plt.plot(subset.index, subset[rolling_cols[0]], color='red', label=rolling_cols[0])
        plt.title('原始值与滚动均值（前1000步）')
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存到 {save_path}")
        else:
            plt.show()


# 如果直接运行，包含示例使用模式以供快速测试
if __name__ == "__main__":
    pass