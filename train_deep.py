import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import mlflow
from torch.utils.data import TensorDataset, DataLoader
import feature_engineering as fe
from models_deep import LSTMAutoencoder

def train_lstm_autoencoder(dataset_name='Altamira', num_epochs=10, batch_size=512, lr=0.001):
    print("Loading precomputed centered matrix...")
    df = pd.read_parquet(f"Preprocess/{dataset_name}_NDVI_CenteredMatrix.parquet")
    df = df.replace(-99999, np.nan)
    
    print("Extracting Time-Aware Features...")
    feature_tensor = fe.build_time_aware_features(df.values, window_size=3)
    
    # We replace any remaining NaNs with 0.0 because PyTorch cannot backpropagate through NaNs
    feature_tensor = np.nan_to_num(feature_tensor, nan=0.0)
    
    # Convert numpy array to PyTorch float32 tensor
    x_tensor = torch.tensor(feature_tensor, dtype=torch.float32)
    
    # Create DataLoader to feed GPU in batches
    dataset = TensorDataset(x_tensor, x_tensor) # In an autoencoder, the target is the input itself
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Automatically use CUDA GPU if available!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- USING DEVICE: {device} ---")
    
    model = LSTMAutoencoder(num_features=5, hidden_dim=64, num_layers=2).to(device)
    criterion = nn.MSELoss() # We measure anomaly by Mean Squared Error of reconstruction
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup MLflow
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"DynaLand_{dataset_name}_DeepLearning")
    with mlflow.start_run(run_name="LSTM_Autoencoder_Phase3"):
        mlflow.log_params({
            "model": "LSTMAutoencoder",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "hidden_dim": 64
        })
        
        print("Starting training loop...")
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            t0 = time.time()
            for batch_x, batch_y in dataloader:
                # Move batches to GPU
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                
                # Calculate Reconstruction Error
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            t1 = time.time()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Time: {t1-t0:.2f}s")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
        print("Training complete!")
        # Save model weights locally and log to MLflow
        torch.save(model.state_dict(), "lstm_autoencoder.pth")
        mlflow.log_artifact("lstm_autoencoder.pth")

if __name__ == '__main__':
    train_lstm_autoencoder()
