from collections import OrderedDict
import os
from pathlib import Path
import flwr as fl
from flwr.app import Context
from flwr.client import ClientApp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from shared.fraud_detection_fl.utils import load_parquet_data
from shared.fraud_detection_fl.model import FraudMLP

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, X_train, y_train, input_dim, partition_id, batch_size=1024, epochs=1, enable_mlflow=False):        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FraudMLP(input_dim).to(self.device)
        self.epochs = epochs
        self.partition_id = partition_id
        self.enable_mlflow = enable_mlflow
        
        # Full local dataset
        full_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        )
        
        # 80/20 Train-Validation Split for Client-Side Evaluation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    def get_parameters(self, config):
        # Extract model weights as a list of NumPy arrays.
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # Update local PyTorch model state with global parameters from server.
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Train model locally on client data.
        self.set_parameters(parameters)
        current_round = config.get("server_round", 0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        self.model.train()
        total_train_loss = 0.0
        total_samples = 0

        for epoch in range(self.epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * X_batch.size(0)
                total_samples += y_batch.size(0)

        avg_loss = total_train_loss / total_samples if total_samples > 0 else 0.0

        # Optional Client-Side Local Metrics Reporting
        metrics = {
            "client_train_loss": avg_loss,
            "partition_id": self.partition_id
        }

        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters, config):
        # Evaluate global model on client local test partition.
        self.set_parameters(parameters)
        criterion = nn.BCEWithLogitsLoss()

        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return float(avg_loss), total, {"accuracy": accuracy, "partition_id": self.partition_id}

def client_fn(context: Context):
    # Context automatically injects partition-id assigned by Flower SuperNode engine
    partition_id = context.node_config.get("partition-id", os.getenv("PARTITION_ID", 0))
    data_dir_str = context.run_config.get(
      "data-dir", os.getenv("CLIENT_DATA_DIR", "/app/client_edge/data/processed")
    )
    data_dir = Path(data_dir_str)
    
    batch_size = context.run_config.get("batch-size", 1024)
    epochs = context.run_config.get("local-epochs", 1)

    client_file = data_dir / f"client_{partition_id}.parquet"
    print(f"[ClientApp] Loading local partition data from: {client_file}")
    
    X_train, y_train, input_dim = load_parquet_data(client_file)

    return FlowerClient(
        X_train=X_train,
        y_train=y_train,
        input_dim=input_dim,
        partition_id=partition_id,
        batch_size=batch_size,
        epochs=epochs,
    ).to_client()


# Entry point referenced by pyproject.toml
app = ClientApp(client_fn=client_fn)