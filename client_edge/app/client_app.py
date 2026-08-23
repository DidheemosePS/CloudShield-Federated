from collections import OrderedDict
import os
from pathlib import Path
import flwr as fl
from flwr.app import Context
from flwr.client import ClientApp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared.fraud_detection_fl.utils import load_parquet_data
from shared.fraud_detection_fl.model import FraudMLP

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, X_train, y_train, input_dim, batch_size=1024, epochs=1):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = FraudMLP(input_dim).to(self.device)
        self.epochs = epochs

        # Convert numpy partitions to PyTorch DataLoaders
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        )
        self.train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

    def get_parameters(self, config):
        # Extract model weights as a list of NumPy arrays.
        return [
            val.cpu().numpy() for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):
        # Update local PyTorch model state with global parameters from server.
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Train model locally on client data.
        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(
                    self.device
                )
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Evaluate global model on client local test partition.
        self.set_parameters(parameters)
        criterion = nn.BCEWithLogitsLoss()

        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(
                    self.device
                )
                outputs = self.model(X_batch)
                loss += criterion(outputs, y_batch).item()
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return float(loss), total, {"accuracy": accuracy}

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
        batch_size=batch_size,
        epochs=epochs,
    ).to_client()


# Entry point referenced by pyproject.toml
app = ClientApp(client_fn=client_fn)