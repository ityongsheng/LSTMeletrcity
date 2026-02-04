import torch
import shap
import numpy as np
import pandas as pd
import warnings
from typing import List

# Suppress warnings from SHAP which can be verbose
warnings.filterwarnings("ignore")

class XAIEngine:
    def __init__(self, model: torch.nn.Module, background_data: np.ndarray, feature_names: List[str] = None):
        """
        Args:
            model: PyTorch model.
            background_data: Representative background dataset (numpy) for SHAP initialization.
                             Shape (Samples, Time, Features).
            feature_names: List of feature names matching the last dimension.
        """
        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
        
        # Prepare background data for DeepExplainer or GradientExplainer
        # Converting to tensor
        self.background_tensor = torch.tensor(background_data, dtype=torch.float32).to(self.device)
        
        # We use GradientExplainer (better for PyTorch) or DeepExplainer.
        # DeepExplainer is often preferred for deep networks.
        try:
            self.explainer = shap.DeepExplainer(self.model, self.background_tensor)
        except Exception as e:
            print(f"DeepExplainer init failed ({e}), falling back to GradientExplainer")
            self.explainer = shap.GradientExplainer(self.model, self.background_tensor)

    def explain_local(self, input_sample: np.ndarray):
        """
        Generate SHAP values for a single input sequence.
        Args:
            input_sample: (1, Time, Features) or (Time, Features)
        Returns:
            shap_values, expected_value
        """
        self.model.eval()
        
        if input_sample.ndim == 2:
            input_sample = input_sample[np.newaxis, ...]
            
        input_tensor = torch.tensor(input_sample, dtype=torch.float32).to(self.device)
        
        # Compute SHAP values
        # shap_values is a list for multi-output, tensor/array for single output
        shap_values = self.explainer.shap_values(input_tensor)
        
        # If output is single dim, it might come wrapped
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # shap_values shape: (1, Time, Features)
        
        return shap_values

    def get_feature_importance(self, shap_values: np.ndarray):
        """
        Calculate global feature importance by averaging absolute SHAP values.
        """
        # avg over samples and time
        # shap_values: (Samples, Time, Features)
        
        if shap_values.ndim == 3:
            global_imp = np.abs(shap_values).mean(axis=(0, 1))
        else:
            global_imp = np.abs(shap_values).mean(axis=0)
            
        if self.feature_names:
            return dict(zip(self.feature_names, global_imp))
        return global_imp

    def explain_in_text(self, shap_values, sample_idx=0):
        """
        Simple heuristic to explain prediction in text.
        """
        # Take the mean SHAP contribution of each feature over the time window
        # shap_values: (1, T, F) -> mean -> (F,)
        if isinstance(shap_values, list): # Handle DeepExplainer output quirk
            sv = shap_values[0]
        else:
            sv = shap_values
            
        if sv.ndim == 3:
            contributions = sv[sample_idx].mean(axis=0)
        else:
            contributions = sv.mean(axis=0) # fallback
            
        if self.feature_names is None:
            return "Feature names not provided, cannot generate text explanation."
            
        # Top 2 positive and Top 2 negative contributors
        sorted_idx = np.argsort(contributions)
        top_pos = sorted_idx[-2:][::-1] # indices of highest pos
        top_neg = sorted_idx[:2]       # indices of lowest neg
        
        explanation = []
        
        # Positive drivers
        pos_reasons = []
        for idx in top_pos:
            if contributions[idx] > 0:
                pos_reasons.append(f"{self.feature_names[idx]} (val={contributions[idx]:.4f})")
        
        if pos_reasons:
            explanation.append(f"Prediction is pushed HIGHER mainly by: {', '.join(pos_reasons)}.")
            
        # Negative drivers
        neg_reasons = []
        for idx in top_neg:
            if contributions[idx] < 0:
                neg_reasons.append(f"{self.feature_names[idx]} (val={contributions[idx]:.4f})")
                
        if neg_reasons:
            explanation.append(f"Prediction is pulled LOWER mainly by: {', '.join(neg_reasons)}.")
            
        return " ".join(explanation)

