import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
import joblib
import mlflow.pytorch
from mlflow import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import torch
import torch.nn.functional as F

# Configure MLflow tracking URI to point to your cluster service
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8080"))

client = MlflowClient()

model_name = "FraudDetection_GlobalModel"
model_version_alias = "champion"

# Global container or app state container for the model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Get information about the model
        model_info = client.get_model_version_by_alias(model_name, model_version_alias)
        run_id = model_info.run_id
        print(f"Loading model version {model_info.version} with alias '{model_version_alias}'")
        
        # Get the model version using a model URI
        model_uri = f"models:/{model_name}@{model_version_alias}"
        
        # Load the PyTorch model artifact and set to evaluation mode
        model = mlflow.pytorch.load_model(model_uri)
        
        # Store in global or app.state dictionary
        ml_models["global_fraud_model"] = model
        
        # Download and load the matching global feature scaler from MLflow artifacts
        scaler_artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessing/global_scaler.pkl"
        )
        ml_models["feature_scaler"] = joblib.load(scaler_artifact_path)
        
        print("Model and global feature scaler successfully loaded into memory.")        
        yield
    except Exception as e:
        print(f"CRITICAL: Startup initialization failed: {e}")
        # Raising an exception here forces the container startup to fail 
        # preventing faulty pods from serving dead traffic.
        raise e
    finally:
        # Clean up memory on shutdown if needed
        ml_models.clear()
        

app = FastAPI(lifespan=lifespan, title="Fraud Detection FL API", version="1.0.0")
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

class PredictionRequest(BaseModel):
    type_CASH_IN: float
    type_CASH_OUT: float
    type_DEBIT: float
    type_PAYMENT: float
    type_TRANSFER: float
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: float
    
class PredictionResponse(BaseModel):
    status: str
    fraud_probability: float
    prediction: int

class RootResponse(BaseModel):
    service: str
    status: str
    version: str
    docs_url: str
    health_check: str

@app.get("/", response_model=RootResponse, tags=["Root"])
def read_root():
    # Root endpoint providing basic service metadata and navigation.
    return {
        "service": "Fraud Detection Federated Learning API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "ready_check": "/ready"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["API Endpoints"])
async def predict(data: PredictionRequest):
    model = ml_models.get("global_fraud_model")
    scaler = ml_models.get("feature_scaler")
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded or unavailable")    
    
    try:
        # Extract features in the exact order the model expects
        features = [
            data.type_CASH_IN,
            data.type_CASH_OUT,
            data.type_DEBIT,
            data.type_PAYMENT,
            data.type_TRANSFER,
            data.amount,
            data.oldbalanceOrg,
            data.newbalanceOrig,
            data.oldbalanceDest,
            data.newbalanceDest,
            data.isFlaggedFraud
        ]
        
        # Apply the pre-fitted global scaler to match training stats
        scaled_features = scaler.transform([features])
        
        # Convert to 2D tensor shape (1, 11)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probability = torch.sigmoid(outputs).item()
            prediction = 1 if probability >= 0.5 else 0
    
        return PredictionResponse(status= "success", fraud_probability=round(probability, 4), prediction=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        

@app.get("/health", status_code=status.HTTP_200_OK, tags=["API Endpoints"])
def health():
    return {"status": "healthy"}

@app.get("/ready", tags=["API Endpoints"])
def ready():
    model = ml_models.get("global_fraud_model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable")        
    return {"status": "success", "model_loaded": True}