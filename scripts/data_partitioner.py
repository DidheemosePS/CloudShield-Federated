from pathlib import Path
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_data_partitioner(
    raw_data_path,
    client_output_dir,
    server_output_dir,
    num_clients,
    alpha,
    seed,
):
    client_path = Path(client_output_dir)
    client_path.mkdir(parents=True, exist_ok=True)
    
    server_path = Path(server_output_dir)
    server_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw dataset from {raw_data_path}")
    # Read raw dataset
    raw_df = pl.read_csv(raw_data_path)
    
    # Remove unnecessary columns (eg: nameOrig)
    cleaned_df = raw_df.drop(["step", "nameOrig", "nameDest"], strict=True)
    # One-Hot Encode categorical 'type' feature
    encoded_df = cleaned_df.to_dummies("type", drop_first=False)
    
    # Separate Features (X) and Target Label (y)
    feature_cols = [col for col in encoded_df.columns if col != "isFraud"]

    X = encoded_df.select(feature_cols).to_numpy()
    y = encoded_df.select("isFraud").to_numpy().ravel()

    # Global Train/Test Split for FL Server Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Scale global server test set using global training stats
    global_scaler = StandardScaler()
    X_train_scaled = global_scaler.fit_transform(X_train)
    X_test_scaled = global_scaler.transform(X_test)
    
    # Save Server Test Partition to Parquet
    test_df = pl.DataFrame(X_test_scaled, schema=feature_cols).with_columns(
        pl.Series("isFraud", y_test)
    )
    
    # Save Server Test Partition to the Server Directory
    test_file = server_path / "server_test.parquet"
    test_df.write_parquet(test_file)
    print(
        f"Saved server test set: {test_file} ({len(test_df):,} samples)"
    )
    
    # Dirichlet Non-IID Partitioning for Clients
    np.random.seed(seed)
    num_classes = len(np.unique(y_train)) # farget label unique values 0 & 1 = 2
    class_proportions = np.random.dirichlet([alpha] * num_clients, size=num_classes) # decideds how to split the data amoung the clients
    client_indices = {i: [] for i in range(num_clients)} # {0: [], 1: [], 2: []}
    
    for c in range(num_classes):
        idx_c = np.where(y_train == c)[0]
        if len(idx_c) == 0:
            continue
        np.random.shuffle(idx_c)
        
        # Calculate sample counts per client
        counts = (class_proportions[c] * len(idx_c)).astype(int)
        counts[-1] = len(idx_c) - np.sum(counts[:-1]) # Fix rounding deficits
        
        splits = np.split(idx_c, np.cumsum(counts)[:-1])
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split)
    
    # Process and Save Individual Client Parquet Files
    for client_id, indices in client_indices.items():
        indices = np.array(indices, dtype=int)
        X_c = X_train[indices]
        y_c = y_train[indices]

        # Local feature scaling (Zero data leakage between clients)
        local_scaler = StandardScaler()
        X_c_scaled = local_scaler.fit_transform(X_c)

        # Build Polars DataFrame and write to disk
        client_df = pl.DataFrame(
            X_c_scaled, schema=feature_cols
        ).with_columns(pl.Series("isFraud", y_c))

        client_file = client_path / f"client_{client_id}.parquet"
        client_df.write_parquet(client_file)

        fraud_count = np.sum(y_c)
        fraud_pct = (fraud_count / len(y_c)) * 100 if len(y_c) > 0 else 0
        print(
            f"Saved client {client_id}: {client_file} | "
            f"Samples: {len(client_df):>7,} | Fraud: {fraud_count:>5} ({fraud_pct:.2f}%)"
        )

if __name__ == "__main__":
    run_data_partitioner(
        raw_data_path="./client_edge/data/raw/raw_dataset.csv",
        client_output_dir="./client_edge/data/processed",
        server_output_dir="./server_k8s/data",
        num_clients=3,
        alpha=0.5,
        seed=42
    )