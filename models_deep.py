import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder: Compresses the sequence
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder: Reconstructs the sequence from the compressed state
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=num_features,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        # x shape: (Batch_Size, Time_Steps, Features)
        
        # 1. Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Take the final hidden state from the last LSTM layer
        last_hidden = hidden[-1].unsqueeze(1) # Shape: (Batch_Size, 1, hidden_dim)
        
        # 2. Repeat the hidden state so the decoder has an input for every time step
        time_steps = x.size(1)
        repeated_hidden = last_hidden.repeat(1, time_steps, 1) # Shape: (Batch, Time, hidden_dim)
        
        # 3. Decode
        reconstructed_x, _ = self.decoder(repeated_hidden)
        
        return reconstructed_x
