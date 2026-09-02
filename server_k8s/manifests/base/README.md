# Kubernetes Manifests

> **Prerequisite:** Before applying any workloads in this directory, you must generate the required **ConfigMap** and **Secret** containing the TLS certificates created in the `certificates/` folder.

This directory contains the Kubernetes manifests to deploy **Flower SuperLink** alongside **Envoy Gateway** with mTLS re-encryption and SuperNode authentication enabled.

---

## 🛠️ Required Resources Checklist

Before running `kubectl apply -f .`, ensure the following Kubernetes objects are provisioned:

| Resource Name         | Type        | Source Files                                           | Namespace                    | Usage                                                 |
| :-------------------- | :---------- | :----------------------------------------------------- | :--------------------------- | :---------------------------------------------------- |
| `superlink-ca-cert`   | `ConfigMap` | `certificates/ca.crt`                                  | `default` (or deployment NS) | Mounted by Envoy / SuperLink for CA validation        |
| `superlink-tls-certs` | `Secret`    | `certificates/server.crt`<br>`certificates/server.key` | `default` (or deployment NS) | Mounted by SuperLink & Envoy Gateway for internal TLS |

---

## 🚀 Pre-Deployment Setup Steps

Execute these commands from the **workspace root** (where the `certificates/` folder lives) before applying your manifests.

### Step 1: Create the CA ConfigMap

Create a ConfigMap containing `ca.crt`. This allows Envoy Gateway's `BackendTLSPolicy` and SuperLink to trust internal certificate chains:

```bash
kubectl create configmap superlink-ca-cert \
  --from-file=ca.crt=certificates/ca.crt
```

### Step 2: Create the TLS Secret

Create the Kubernetes Secret containing the server certificate (`server.crt`) and private key (`server.key`):

```bash
kubectl create secret tls superlink-tls-certs \
  --cert=certificates/server.crt \
  --key=certificates/server.key
```

### Step 3: Verify Pre-requisite Creation

Confirm that both resources exist in your target namespace before proceeding:

```bash
kubectl get configmap superlink-ca-cert
kubectl get secret superlink-tls-certs
```

### Step 4: Deploying Manifests

Once the ConfigMap and Secret are present, apply the Kubernetes manifests either all at once or in sequence:

**Option A: Apply all manifests at once**

```bash
kubectl apply -f .

**Option B: Apply manifests in sequence**

# 1. Apply SuperLink Deployment and Service
kubectl apply -f superlink-deployment.yaml
kubectl apply -f superlink-service.yaml

# 2. Apply Envoy Gateway and BackendTLSPolicy configs
kubectl apply -f gateway.yaml
kubectl apply -f backend-tls-policy.yaml
```

---

## 🔄 Updating Certificates

If you regenerate certificates in the `certificates/` directory, update the cluster resources and trigger a rolling restart:

```bash
# Update ConfigMap & Secret
kubectl create configmap superlink-ca-cert \
  --from-file=ca.crt=certificates/ca.crt \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret tls superlink-tls-certs \
  --cert=certificates/server.crt \
  --key=certificates/server.key \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart SuperLink deployment to pick up new certificates
kubectl rollout restart deployment/superlink
```
