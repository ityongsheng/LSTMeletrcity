import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models import LSTMModel
from typing import Tuple, Optional

class EnergyForecaster:
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, X_train, y_train, epochs=10, batch_size=32, verbose=True):
        """
        Train the LSTM model.
        Args:
            X_train: Numpy array (Samples, Time, Features)
            y_train: Numpy array (Samples, Horizon)
        """
        self.model.train()
        
        # Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                
        return loss_history

    def predict(self, X_test, n_samples=50):
        """
        Predict with uncertainty quantification (MC Dropout).
        
        Returns:
            mean: (batch, output_dim)
            lower_ci: (batch, output_dim) - 2 std dev
            upper_ci: (batch, output_dim) + 2 std dev
        """
        self.model.train() # Important: Keep dropout active!
        
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        mean_pred, std_pred, _ = self.model.predict_with_uncertainty(X_tensor, n_samples=n_samples)
        
        mean_pred = mean_pred.cpu().numpy()
        std_pred = std_pred.cpu().numpy()
        
        # 95% Confidence Interval (approx 2 sigma)
        lower_ci = mean_pred - 2 * std_pred
        upper_ci = mean_pred + 2 * std_pred
        
        return mean_pred, lower_ci, upper_ci

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")
