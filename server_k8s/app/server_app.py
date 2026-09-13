from collections import OrderedDict
import os
from pathlib import Path
import flwr as fl
from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared.fraud_detection_fl.model import FraudMLP
from shared.fraud_detection_fl.utils import load_parquet_data

client = MlflowClient()

# Server-side evaluation on global test dataset
def get_evaluate_fn(test_data_path: Path):
    test_file = test_data_path / "server_test.parquet"
    X_test, y_test, input_dim = load_parquet_data(test_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )
    torch.set_num_threads(2)
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=0
    )
    best_pr_auc = 0.0

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        nonlocal best_pr_auc
        model = FraudMLP(input_dim).to(device)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        criterion = nn.BCEWithLogitsLoss()
        total_loss, all_preds, all_targets = 0.0, [], []

        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
            
                loss = criterion(logits, y_b)
                total_loss += loss.item() * y_b.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(y_b.cpu().numpy())

        all_preds = np.array(all_preds).ravel()
        all_targets = np.array(all_targets).ravel()

        binary_preds = (all_preds >= 0.5).astype(int)
        f1 = float(f1_score(all_targets, binary_preds, zero_division=0))
        precision, recall, _ = precision_recall_curve(all_targets, all_preds)
        pr_auc = float(auc(recall, precision))
        avg_loss = float(total_loss / len(test_loader))

        print(
            f"[Server Round {server_round}] Test Loss: {avg_loss:.4f} | F1-Score: {f1:.4f} | PR-AUC: {pr_auc:.4f}"
        )

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "server_test_loss": avg_loss,
                    "server_test_f1": f1,
                    "server_test_pr_auc": pr_auc,
                },
                step=server_round,
            )

            # Checkpoint global model whenever PR-AUC reaches a new peak
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                mlflow.log_metric("best_pr_auc", best_pr_auc, step=server_round)
                try:
                    input_example = X_test[:2].astype(np.float32)
                    dummy_output = model(torch.tensor(input_example).to(device)).detach().cpu().numpy()
                    signature = infer_signature(input_example, dummy_output)

                    # Log model with cloudpickle serialization and signature
                    model_info = mlflow.pytorch.log_model(
                        pytorch_model=model,
                        name="best_global_model",
                        step=server_round,
                        signature=signature,
                        input_example=input_example,
                        registered_model_name="FraudDetection_GlobalModel",
                    )
                    
                    # Log the global scaler artifact in the same run
                    scaler_path = test_data_path / "global_scaler.pkl"
                    if scaler_path.exists():
                        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
        
                    # Assign the 'champion' alias to this specific peak version
                    client.set_registered_model_alias(
                        name="FraudDetection_GlobalModel",
                        alias="champion",
                        version=model_info.registered_model_version
                    )
                    
                    print(f"[MLflow] Logged champion model & global scaler (PR-AUC: {best_pr_auc:.4f})")
                except Exception as e:
                    print(f"[MLflow] Model artifact upload failed: {e}")

        return avg_loss, {"f1": f1, "pr_auc": pr_auc}

    return evaluate


# Aggregates metrics reported by SuperNodes, weighted by sample count
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
      return {} 
  
    weighted_f1 = (
        sum(num_examples * float(m.get("f1", 0.0)) for num_examples, m in metrics)
        / total_examples
    )
    weighted_pr_auc = (
        sum(
            num_examples * float(m.get("pr_auc", 0.0))
            for num_examples, m in metrics
        )
        / total_examples
    )   
    
    return {"f1": weighted_f1, "pr_auc": weighted_pr_auc}


# Custom strategy to clean up MLflow active run lifecycle
class MLflowFedAvg(FedAvg):
    def aggregate_evaluate(self, server_round: int, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Log aggregated metrics with explicit step mapping
        if mlflow.active_run() and aggregated_metrics:
            if "f1" in aggregated_metrics:
                mlflow.log_metric("aggregated_client_f1", aggregated_metrics["f1"], step=server_round)
            if "pr_auc" in aggregated_metrics:
                mlflow.log_metric("aggregated_client_pr_auc", aggregated_metrics["pr_auc"], step=server_round)
                
        return aggregated_loss, aggregated_metrics

def server_fn(context: Context):
    num_rounds = context.run_config.get("num-server-rounds", 5) 
    
    # Explicit container-native data paths
    test_data_path = Path(
        os.getenv("TEST_DATA_PATH", "/app/server_k8s/data")
    )   
    
    # MLflow tracking URI defaults to local sqlite mount or K8s Service DNS
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8080")
    mlflow.set_tracking_uri(mlflow_uri)
    
    experiment_name = "Federated-Fraud-Detection"
    client = mlflow.MlflowClient()
    
    # Check if experiment exists and handle soft-deleted state
    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None:
        if exp.lifecycle_stage == "deleted":
            print(f"[MLflow] Experiment '{experiment_name}' is soft-deleted. Restoring...")
            client.restore_experiment(exp.experiment_id)
        mlflow.set_experiment(experiment_name)
    else:
        mlflow.create_experiment(experiment_name)
    
    # Clean active run management in server initialization
    if mlflow.active_run():
        mlflow.end_run()
    
    active_run = mlflow.start_run(run_name="flower_round_execution")
    print(f"[MLflow] Active run initialized: {active_run.info.run_id}")
    
    mlflow.log_params({
        "num_rounds": num_rounds,
        "batch_size": 1024,
        "local_epochs": 1,
        "strategy": "FedAvg",
        "min_fit_clients": 3,
        "min_available_clients": 3,
    })
    
    # Use MLflowFedAvg instead of standard FedAvg
    strategy = MLflowFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(test_data_path),
    )   

    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)    

# Entry point referenced by pyproject.toml
app = ServerApp(server_fn=server_fn)