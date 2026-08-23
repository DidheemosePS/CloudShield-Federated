from pathlib import Path
import polars as pl

# Load parquet data
def load_parquet_data(file_path: Path):
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {file_path}. Please run data_partitioner.py first!"
        )

    df = pl.read_parquet(file_path)
    feature_cols = [col for col in df.columns if col != "isFraud"]

    X = df.select(feature_cols).to_numpy()
    y = df.select("isFraud").to_numpy().ravel()
    input_dim = len(feature_cols)

    return X, y, input_dim

