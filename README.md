# CloudShield-Federated: Production-Grade Hybrid Federated Learning MLOps Platform (Version 1.0)

**CloudShield-Federated** is an enterprise-ready, cross-silo Federated Learning platform engineered for secure financial fraud detection. It bridges privacy-preserving distributed learning at the edge with centralized cloud orchestration, experiment tracking, secure zero-trust communication, real-time inference serving, and cluster-wide telemetry.

---

## 🏗️ System Architecture Overview

The platform uses a **hybrid split-architecture** designed to enforce absolute data privacy across client silos while maintaining centralized cloud governance:

<p align="center">
  <img src="assets/custom-architecture.png" alt="Custom Federated Learning Architecture" width="800">
</p>

---

## 📂 Monorepo Structure

```text
cloudshield-federated/
├── .github/                      # GitHub
├── assets/
│   └── custom-architecture.png   # System architecture diagram
├── certificates/                 # TLS certs, configuration, and keys (ca, server, san)
├── client_edge/                  # Edge client node implementation
│   ├── app/                      # Client app logic and python cache
│   ├── data/                     # Local partitioned datasets (raw & processed parquets)
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── pyproject.toml
├── keys/                         # Supernode cryptographic keys
├── scripts/
│   └── data_partitioner.py       # Dataset partitioning utility
├── server_k8s/                   # Central server and Kubernetes MLOps stack
│   ├── app/                      # Server orchestrator & FastAPI prediction service
│   ├── data/                     # Global data scalers and test sets
│   ├── manifests/                # K8s base manifests (Envoy Gateway, Grafana, MLflow, Prometheus, Flower etc.)
│   ├── mlflow_data/              # Local MLflow tracking data, sqlite db, and model artifacts
│   ├── Dockerfile
│   ├── Dockerfile.prediction     # Dedicated Dockerfile for the inference API
│   └── pyproject.toml
├── shared/                       # Shared FL logic across client and server
│   ├── fraud_detection_fl/       # Shared neural network model and utilities
│   └── pyproject.toml            # Shared package configuration
├── Dockerfile                    # Root container definition
├── LICENCE
├── README.md
├── docker-compose.yaml           # Local multi-container deployment
├── pyproject.toml                # Root project dependencies configuration
└── uv.lock                       # uv dependency lock file

```

---

## 🚀 Execution Roadmap & Progress

### Phase 1: Local Baseline & Training Optimization (**COMPLETED**)

- **Decoupled Architecture:** Validated Flower 1.x `SuperLink`, `ServerApp`, and `SuperNode` runtime loops over isolated Docker networks.
- **CPU Vectorization:** Tuned batch sizes (`batch_size=1024`) and local epochs (`local_epochs=1`), dropping local iteration overhead by **15x** and preventing server stalls.
- **Data Skew Control:** Implemented sample-weighted metric aggregation (`weighted_average`) to balance non-IID partitions across client nodes.

### Phase 2: Experiment Tracking & Model Checkpointing (**COMPLETED**)

- **MLflow Integration:** Deployed tracking server backend (`http://mlflow:5000`) logging round-by-round training loss, precision, recall, F1-score, and PR-AUC.
- **Safe Checkpointing:** Configured automatic model checkpointing using PyTorch `pt2` graph serialization to save global weights whenever PR-AUC peaks.

### Phase 3: Kubernetes Infrastructure, Envoy Gateway TLS & Node Auth (**COMPLETED**)

- **Envoy Gateway Ingress:** Implemented edge TLS termination on port 443 with backend re-encryption to protect cluster-internal gRPC communications.
- **Cryptographic Node Whitelisting:** Enforced `--enable-supernode-auth` on `SuperLink` using Elliptic Curve key pairs (`.pem`/`.pub`) registered via `flwr supernode register`. Only verified clients are allowed to join federated training rounds.
- **Persistent State:** Attached a Kubernetes PersistentVolume to `SuperLink` to ensure registered client node whitelists survive pod evictions and rolling restarts.

### Phase 4: Central Inference, Serving & Telemetry (**COMPLETED - VERSION 1.0**)

- **FastAPI Lifespan Management:** Deployed `prediction_app.py` inside Kubernetes utilizing asynchronous context management to fetch the `champion` model alias and pre-fitted `global_scaler.pkl` directly from MLflow upon startup.
- **Input Validation & Safety:** Integrated strict Pydantic schemas (`PredictionRequest`, `PredictionResponse`) to validate incoming transaction payloads and prevent malformed requests.
- **Probes & Telemetry Integration:** Configured lightweight Kubernetes liveness (`/health`) and readiness (`/ready`) probes tied to model availability, alongside automatic metric collection via `prometheus-fastapi-instrumentator` at `/metrics`.
- **Observability Platform:** Deployed and connected Prometheus servers and Grafana dashboards to visualize real-time request throughput, P95/P99 latency percentiles, and error metrics.

---

## 🔮 Future Phases & Roadmap

### Phase 5: Autonomous Diagnostic Agents & LLM Observability (**PLANNED / VERSION 2.0**)

- **Autonomous Log Ingestion:** Integrate an automated log-parsing agent to ingest streamed pod metrics and identify non-IID data drift or straggling node connections in real time.
- **Automated Remediation:** Implement automated scaling triggers for edge resources based on federated training round durations.
