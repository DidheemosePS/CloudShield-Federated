# Certificates Directory

> **Note:** Sensitive TLS certificates and private keys are excluded from version control (`.gitignore`) to prevent security leaks.

This directory holds the **TLS/SSL certificates** used to secure gRPC communications between **SuperLink**, **Envoy Gateway**, **SuperNodes**, and the **Flower CLI**.

---

## 🛠️ Required Files & Configuration

Before starting the services via Docker Compose or Kubernetes, this folder must contain:

| File / Artifact | Description                                                          | Who Needs It                              |
| :-------------- | :------------------------------------------------------------------- | :---------------------------------------- |
| `san.cnf`       | OpenSSL configuration file defining Subject Alternative Names (SANs) | Certificate generation process            |
| `ca.crt`        | Root Certificate Authority (CA) certificate                          | All Clients, SuperNodes, SuperLink, Envoy |
| `ca.key`        | Private key for the Certificate Authority                            | Certificate generation process only       |
| `server.crt`    | Server TLS Certificate (generated using `san.cnf`)                   | SuperLink & Envoy Gateway                 |
| `server.key`    | Private key for Server Certificate                                   | SuperLink & Envoy Gateway                 |

---

## 🚀 Generating Certificate Files

When setting up the environment locally, ensure `san.cnf` is configured correctly in this directory, then run the generation commands below from within the `certificates/` folder.

### Commands to Generate Certificates

```bash
# 1. Generate Root CA Private Key
openssl genrsa -out ca.key 4096

# 2. Generate Root CA Certificate (ca.crt)
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 \
  -out ca.crt \
  -subj "/C=IE/ST=Dublin/L=Dublin/O=LocalDev/CN=LocalDevRootCA"

# 3. Generate Server Private Key (server.key)
openssl genrsa -out server.key 2048

# 4. Generate Certificate Signing Request (CSR)
openssl req -new -key server.key \
  -out server.csr \
  -config san.cnf

# 5. Sign Server Certificate (server.pem) using Root CA and san.cnf extensions
openssl x509 -req -in server.csr \
  -CA ca.crt \
  -CAkey ca.key \
  -CAcreateserial \
  -out server.crt \
  -days 365 \
  -sha256 \
  -extfile san.cnf \
  -extensions req_ext

# 6. Verify signing chain (Must output: server.pem: OK)
openssl verify -CAfile ca.crt server.pem

# 7. Verify SAN entries in generated server certificate
openssl x509 -in server.pem -noout -text | grep -A 2 "Subject Alternative Name"

# 8. Clean up temporary CSR and serial tracking files
rm -f server.csr ca.srl
```
