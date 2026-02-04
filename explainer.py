import torch
import shap
import numpy as np
import pandas as pd
import warnings
from typing import List

# 抑制SHAP的冗长警告信息
warnings.filterwarnings("ignore")


class XAIEngine:
    def __init__(self, model: torch.nn.Module, background_data: np.ndarray, feature_names: List[str] = None):
        """
        初始化XAI引擎

        参数：
            model: PyTorch模型
            background_data: SHAP初始化所需的代表性背景数据集(numpy数组)，形状为(样本数, 时间步长, 特征数)
            feature_names: 与最后一个维度匹配的特征名称列表
        """
        self.model = model
        self.feature_names = feature_names  # 特征名称列表
        self.device = next(model.parameters()).device  # 获取模型所在的设备（CPU或GPU）

        # 准备DeepExplainer或GradientExplainer所需的背景数据
        # 将背景数据转换为张量并移动到对应设备
        self.background_tensor = torch.tensor(background_data, dtype=torch.float32).to(self.device)

        # 尝试使用DeepExplainer（深度网络首选），失败则回退到GradientExplainer
        try:
            self.explainer = shap.DeepExplainer(self.model, self.background_tensor)
        except Exception as e:
            print(f"DeepExplainer初始化失败 ({e})，回退到GradientExplainer")
            self.explainer = shap.GradientExplainer(self.model, self.background_tensor)

    def explain_local(self, input_sample: np.ndarray):
        """
        为单个输入序列生成SHAP值

        参数：
            input_sample: 输入序列，形状为(1, 时间步长, 特征数) 或 (时间步长, 特征数)

        返回：
            shap_values: SHAP值
            expected_value: 期望值（基础值）
        """
        self.model.eval()  # 将模型设置为评估模式

        # 确保输入为3维：添加样本维度
        if input_sample.ndim == 2:
            input_sample = input_sample[np.newaxis, ...]

        # 将输入转换为张量并移动到设备
        input_tensor = torch.tensor(input_sample, dtype=torch.float32).to(self.device)

        # 计算SHAP值
        # shap_values对于多输出是列表，对于单输出是张量/数组
        shap_values = self.explainer.shap_values(input_tensor)

        # 如果输出是单维度，可能需要解包
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # 取第一个输出

        # shap_values形状: (1, 时间步长, 特征数)

        return shap_values

    def get_feature_importance(self, shap_values: np.ndarray):
        """
        通过平均绝对SHAP值计算全局特征重要性

        参数：
            shap_values: SHAP值数组

        返回：
            如果提供了feature_names，返回特征重要性字典，否则返回重要性数组
        """
        # 在样本和时间维度上取平均
        # shap_values形状: (样本数, 时间步长, 特征数)

        if shap_values.ndim == 3:
            # 计算绝对值的平均值（跨样本和时间步长）
            global_imp = np.abs(shap_values).mean(axis=(0, 1))
        else:
            # 如果只有2维，则在样本维度取平均
            global_imp = np.abs(shap_values).mean(axis=0)

        # 如果提供了特征名称，返回字典，否则返回数组
        if self.feature_names:
            return dict(zip(self.feature_names, global_imp))
        return global_imp

    def explain_in_text(self, shap_values, sample_idx=0):
        """
        使用启发式方法生成文本解释

        参数：
            shap_values: SHAP值数组
            sample_idx: 样本索引（默认为0）

        返回：
            文本形式的解释
        """
        # 取每个特征在时间窗口上的平均SHAP贡献
        # shap_values形状: (1, 时间步长, 特征数) -> 平均 -> (特征数,)
        if isinstance(shap_values, list):  # 处理DeepExplainer的输出特性
            sv = shap_values[0]
        else:
            sv = shap_values

        # 计算每个特征的平均贡献
        if sv.ndim == 3:
            contributions = sv[sample_idx].mean(axis=0)  # 时间维度取平均
        else:
            contributions = sv.mean(axis=0)  # 备用方案

        # 如果没有特征名称，无法生成文本解释
        if self.feature_names is None:
            return "未提供特征名称，无法生成文本解释。"

        # 找出前2个正向贡献者和前2个负向贡献者
        sorted_idx = np.argsort(contributions)  # 按贡献值排序
        top_pos = sorted_idx[-2:][::-1]  # 正向贡献最大的2个特征索引（从大到小）
        top_neg = sorted_idx[:2]  # 负向贡献最大的2个特征索引（从小到大）

        explanation = []  # 存储解释文本

        # 正向驱动因素
        pos_reasons = []
        for idx in top_pos:
            if contributions[idx] > 0:  # 只处理正向贡献
                pos_reasons.append(f"{self.feature_names[idx]} (值={contributions[idx]:.4f})")

        if pos_reasons:
            explanation.append(f"预测值主要被以下因素推高: {', '.join(pos_reasons)}.")

        # 负向驱动因素
        neg_reasons = []
        for idx in top_neg:
            if contributions[idx] < 0:  # 只处理负向贡献
                neg_reasons.append(f"{self.feature_names[idx]} (值={contributions[idx]:.4f})")

        if neg_reasons:
            explanation.append(f"预测值主要被以下因素拉低: {', '.join(neg_reasons)}.")

        return " ".join(explanation)  # 连接所有解释部分