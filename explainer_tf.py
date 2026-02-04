import shap
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt


class ExplainerTF:
    def __init__(self, model, background_data):
        """
        初始化TensorFlow模型的可解释性工具

        参数：
            model: 训练好的Keras模型
            background_data: 用于摘要的背景数据集，Numpy数组，形状为(样本数, 时间步长, 特征数)
        """
        self.model = model
        # 存储原始形状以便在包装器中重新塑形
        self.input_shape = background_data.shape[1:]  # (时间步长, 特征数)

        # 为通用解释器(Permutation/Kernel)展平背景数据，这些解释器通常偏好2D表格数据
        # 展平后形状: (样本数, 时间步长*特征数)
        self.bg_flat = background_data.reshape(background_data.shape[0], -1)

        # 初始化解释器
        # 首先尝试DeepExplainer(原生TF)，失败则回退到PermutationExplainer(模型无关)
        # 强制回退: DeepExplainer在当前环境的TF 2.x急切执行模式下不稳定
        try:
            # 显式抛出错误以强制回退流程
            raise RuntimeError("为TF2稳定性强制使用PermutationExplainer")
            # self.explainer = shap.DeepExplainer(model, background_data)
            # self.mode = 'deep'
        except Exception:
            print("DeepExplainer失败。使用带有扁平化输入的PermutationExplainer。")

            # 包装函数：将2D扁平输入重塑回3D以供模型使用
            def predict_wrapper(x_flat):
                # x_flat形状: (样本数, 时间步长*特征数)
                x_3d = x_flat.reshape(x_flat.shape[0], *self.input_shape)
                # 直接使用model()以避免在此上下文中使用tf.data的开销/错误
                # 如果需要则转换为张量，但Keras通常能处理numpy数组
                return model(x_3d).numpy()

            # 在扁平化数据上使用Independent掩码器
            masker = shap.maskers.Independent(self.bg_flat)
            self.explainer = shap.PermutationExplainer(predict_wrapper, masker)
            self.mode = 'permutation'

    def compute_shap_values(self, X_sample):
        """
        为给定样本或批次计算SHAP值

        返回：
            SHAP值数组
        """
        if self.mode == 'deep':
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                return shap_values[0]  # 如果是列表，取第一个输出
            return shap_values

        elif self.mode == 'permutation':
            # 展平输入
            X_flat = X_sample.reshape(X_sample.shape[0], -1)

            # 运行解释器，设置较高的最大评估次数
            # 返回Explanation对象
            explanation = self.explainer(X_flat, max_evals=2000)

            # 获取值并重塑回3D形状(样本数, 时间步长, 特征数)
            values_flat = explanation.values

            # 处理列表输出(多输出情况)
            if isinstance(values_flat, list):
                values_flat = values_flat[0]

            # 重塑回(样本数, 时间步长, 特征数)
            shap_values = values_flat.reshape(X_sample.shape[0], *self.input_shape)

            # 存储base_values供后续导出使用(临时附加)
            self.last_base_value = explanation.base_values

            return shap_values

    def get_base_value(self):
        """
        获取基础值(期望值)

        返回：
            基础值(浮点数)
        """
        if self.mode == 'deep':
            # DeepExplainer的expected_value
            bv = self.explainer.expected_value
            if isinstance(bv, list):
                return bv[0]  # 如果是列表，取第一个
            return bv
        else:
            # PermutationExplainer通常在Explanation对象中返回
            # 我们在上一次调用中存储了它
            if hasattr(self, 'last_base_value'):
                bv = self.last_base_value
                # 如果是数组，取其平均值
                if isinstance(bv, (np.ndarray, list)):
                    return np.mean(bv)  # 每个样本的基础值？通常是常数
                return bv
            return 0.0  # 未知情况

    def plot_summary(self, shap_values, features, feature_names=None, save_path=None):
        """
        生成通用摘要图

        参数：
            shap_values: SHAP值数组
            features: 与shap_values对应的实际输入值
            feature_names: 特征名称列表
            save_path: 保存图片的路径(可选)
        """
        # SHAP摘要图通常期望2D输入(样本数, 特征数)
        # LSTM输入是3D的(样本数, 时间步长, 特征数)
        # 我们需要展平或在时间维度上聚合以正确可视化特征重要性
        # 策略：按特征计算时间维度上的平均绝对SHAP值

        # shap_values形状: (样本数, 时间步长, 特征数)
        if shap_values.ndim == 3:
            # 在时间维度上聚合 -> (样本数, 特征数)
            shap_values_2d = np.abs(shap_values).mean(axis=1)  # 平均影响大小
            features_2d = features.mean(axis=1)  # 特征值的平均值
        else:
            shap_values_2d = shap_values
            features_2d = features

        plt.figure()
        shap.summary_plot(shap_values_2d, features_2d, feature_names=feature_names, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def export_to_json(self, shap_values, base_value, features, feature_names, output_path='shap_data.json'):
        """
        将SHAP数据导出为JSON格式，供前端力导向图使用

        参数：
            shap_values: SHAP值数组，形状为(样本数, 时间步长, 特征数)或(时间步长, 特征数)
            base_value: 解释器的期望值
            features: 输入特征值
            feature_names: 特征名称列表
            output_path: 输出JSON文件路径(默认'shap_data.json')
        """
        # 如果传入的是批次，取第一个样本
        if shap_values.ndim == 3:
            sv = shap_values[0]  # (时间步长, 特征数)
            feat = features[0]
        else:
            sv = shap_values
            feat = features

        # 对于力导向图，我们通常可视化每个特征对该样本预测的"全局"贡献
        # 对于时间序列，我们通常对时间维度求和或取平均
        # 这里我们选择对时间维度求和，以显示"特征X对预测的总贡献"

        # 在时间轴上求和
        sv_agg = sv.sum(axis=0)  # (特征数, )
        feat_agg = feat.mean(axis=0)  # 特征在时间窗口上的平均值

        data = {
            "base_value": float(base_value),
            "shap_values": sv_agg.tolist(),
            "features": feat_agg.tolist(),
            "feature_names": feature_names
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"SHAP数据已导出到 {output_path}")