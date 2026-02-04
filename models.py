import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True -> (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer (Explicitly used for MC Dropout in fully connected part if needed, 
        # but nn.LSTM dropout only affects connections between layers, not the last output)
        self.dropout = nn.Dropout(dropout)
        
        # Fully Connected Layer to map to output
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        # out[:, -1, :] shape: (batch, hidden_dim)
        last_step_out = out[:, -1, :]
        
        # Apply dropout explicitly to last step for MC Dropout effect during inference look-alikes
        last_step_out = self.dropout(last_step_out)
        
        # Linear layer
        prediction = self.fc(last_step_out)
        return prediction

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        Perform Monte Carlo Dropout Inference.
        """
        self.train() # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(self.forward(x).unsqueeze(0))
        
        # Shape: (n_samples, batch, output_dim)
        predictions = torch.cat(predictions, dim=0)
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred, predictions
