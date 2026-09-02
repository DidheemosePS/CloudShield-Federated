Here is the comprehensive, enterprise-grade project documentation (`README.md`) updated to integrate **Phase 4 (Central Inference & Production Telemetry)**.

---

# CloudShield-Federated: Production-Grade Hybrid Federated Learning MLOps Platform

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
fraud-detection-fl/
├── Dockerfile                  # Root container configuration
├── docker-compose.yaml         # Multi-service local orchestration
├── pyproject.toml              # Global project dependencies and build system
├── uv.lock                     # Lockfile for reproducible environment builds
├── LICENCE                     # Platform licensing information
├── README.md                   # Comprehensive project documentation
├── certificates/               # Mutual TLS (mTLS) and CA authority certificates
├── keys/                       # Elliptic-curve cryptographic keys for SuperNode auth
├── scripts/                    # Data partitioning and maintenance utilities
├── shared/                     # Shared PyTorch models, utilities, and schemas
├── client_edge/                # Edge node container runtimes & local data partitions
│   ├── Dockerfile              # Lightweight SuperNode container build
│   ├── docker-compose.yaml     # Edge-specific multi-container configurations
│   ├── pyproject.toml          # Client-side dependency specifications
│   ├── app/                    # Client app logic (ClientApp training loops)
│   └── data/                   # Raw CSV datasets and processed Parquet splits
└── server_k8s/                 # Central Cloud Control Plane & Kubernetes manifests
    ├── Dockerfile              # Central build context for SuperLink/FastAPI
    ├── pyproject.toml          # Server-side dependency specifications
    ├── app/                    # Central services (ServerApp & FastAPI prediction service)
    ├── data/                   # Central server evaluation datasets
    ├── manifests/              # Kubernetes base manifests, PVCs, and Envoy gateways
    └── mlflow_data/            # SQLite tracking database and versioned model artifacts (.pt2)

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

- **Envoy Gateway Ingress:** Implemented edge TLS termination with backend re-encryption to protect cluster-internal gRPC communications.
- **Cryptographic Node Whitelisting:** Enforced `--enable-supernode-auth` on `SuperLink` using Elliptic Curve key pairs (`.pem`/`.pub`) registered via `flwr supernode register`. Only verified clients are allowed to join federated training rounds.

### Phase 4: Central Inference, Serving & Telemetry (**CURRENT**)

- **FastAPI Prediction Service:** Deployed `prediction_app.py` inside the Kubernetes cluster to load peak global checkpoints from MLflow and serve low-latency `/v1/predict` HTTP requests.
- **Production Telemetry:** Configured Prometheus and Grafana scrapers monitoring pod resource usage, inter-node gRPC latency, and client health check heartbeats.

---

## 🔮 Future Phases & Roadmap

### Phase 5: Autonomous Diagnostic Agents & LLM Observability (**PLANNED**)

- **Autonomous Log Ingestion:** Integrate a lightweight diagnostic log-parsing agent to ingest streamed pod metrics and identify non-IID data drift or straggling node connections in real time.
- **Automated Remediation:** Implement automated scaling triggers for edge resources based on federated training round durations.

---
