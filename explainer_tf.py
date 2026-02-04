import shap
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

class ExplainerTF:
    def __init__(self, model, background_data):
        """
        Args:
            model: Trained Keras model.
            background_data: Numpy array (Samples, Time, Features) to summarize.
        """
        self.model = model
        # Store original shape for reshaping in wrapper
        self.input_shape = background_data.shape[1:] # (Time, Features)
        
        # Flatten background for generic explainers (Permutation/Kernel) which prefer 2D tabular
        # (Samples, Time*Features)
        self.bg_flat = background_data.reshape(background_data.shape[0], -1)
        
        # Initialize Explainer
        # Try DeepExplainer (native TF) first, fallback to Permutation (Model Agnostic)
        # FORCE FALLBACK: DeepExplainer is unstable with TF 2.x eager execution in this env.
        # simulating a failure to trigger fallback block
        try:
             # Raise explicit error to force fallback flow
            raise  RuntimeError("Force PermutationExplainer for TF2 stability")
            # self.explainer = shap.DeepExplainer(model, background_data)
            # self.mode = 'deep'
        except Exception:
            print("DeepExplainer failed. Using PermutationExplainer with flattened inputs.")
            
            # Wrapper to reshape 2D flat input back to 3D for model
            def predict_wrapper(x_flat):
                # x_flat: (Samples, Time*Feat)
                x_3d = x_flat.reshape(x_flat.shape[0], *self.input_shape)
                # Use model() directly to avoid tf.data overhead/errors in this context
                # Convert to tensor if needed, but Keras handles numpy usually
                return model(x_3d).numpy()
            
            # Use Independent masker on flattened data
            masker = shap.maskers.Independent(self.bg_flat)
            self.explainer = shap.PermutationExplainer(predict_wrapper, masker)
            self.mode = 'permutation'
            
    def compute_shap_values(self, X_sample):
        """
        Compute SHAP values for a given sample or batch.
        """
        if self.mode == 'deep':
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                return shap_values[0]
            return shap_values
            
        elif self.mode == 'permutation':
            # Flatten input
            X_flat = X_sample.reshape(X_sample.shape[0], -1)
            
            # Run explainer with high max_evals
            # Returns Explanation object
            explanation = self.explainer(X_flat, max_evals=2000)
            
            # Get values and reshape back to 3D (Samples, Time, Feat)
            values_flat = explanation.values
            
            # Handle list output (multi-output)
            if isinstance(values_flat, list):
                values_flat = values_flat[0]
                
            # Reshape back to (Samples, Time, Features)
            shap_values = values_flat.reshape(X_sample.shape[0], *self.input_shape)
            
            # Store base_values for export usage later (hacky attach)
            self.last_base_value = explanation.base_values
            
            return shap_values
            
    def get_base_value(self):
        if self.mode == 'deep':
             # DeepExplainer expected_value
            bv = self.explainer.expected_value
            if isinstance(bv, list): return bv[0]
            return bv
        else:
             # Permutation explainer returns it in Explanation object usually
             # We stored it in last call
             if hasattr(self, 'last_base_value'):
                 bv = self.last_base_value
                 # If array, mean it?
                 if isinstance(bv, (np.ndarray, list)):
                     return np.mean(bv) # Base value per sample? usually constant
                 return bv
             return 0.0 # Unknown
        
    def plot_summary(self, shap_values, features, feature_names=None, save_path=None):
        """
        Generate generic summary plot.
        features: The actual input values corresponding to shap_values.
        """
        # SHAP Summary plot usually expects 2D (Samples, Features).
        # LSTM input is 3D (Samples, Time, Features).
        # We need to flatten or aggregate over time to visualize feature importance properly.
        # Strategy: Mean absolute SHAP over time per feature.
        
        # shap_values shape: (Samples, Time, Features)
        if shap_values.ndim == 3:
            # Aggregate over time dimension -> (Samples, Features)
            shap_values_2d = np.abs(shap_values).mean(axis=1) # Mean impact magnitude
            features_2d = features.mean(axis=1) # Average feature value
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
        Export SHAP data for a specific sample (e.g. the first one) to JSON for frontend Force Plot.
        
        Args:
            shap_values: (Samples, Time, Features) or (Time, Features)
            base_value: The explainer's expected_value.
            features: Input feature values.
            feature_names: List of feature names.
        """
        # Take the first sample if passed a batch
        if shap_values.ndim == 3:
            sv = shap_values[0] # (Time, Features)
            feat = features[0]
        else:
            sv = shap_values
            feat = features
            
        # For force plot, we typically visualize 'Global' contribution per feature for that sample
        # or we pick a specific time step? 
        # Usually for Time Series, we sum contributions over time or pick Mean.
        # Let's Sum them to show "Total contribution of Feature X to the prediction".
        
        # Sum over time axis
        sv_agg = sv.sum(axis=0) # (Features, )
        feat_agg = feat.mean(axis=0) # Mean value of feature over the window
        
        data = {
            "base_value": float(base_value),
            "shap_values": sv_agg.tolist(),
            "features": feat_agg.tolist(),
            "feature_names": feature_names
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"SHAP data exported to {output_path}")
