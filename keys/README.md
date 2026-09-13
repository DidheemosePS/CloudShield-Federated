# Keys Directory (SuperNode Authentication)

> **Note:** Key directory files (`supernode_*` and `supernode_*.pub`) are excluded from version control via `.gitignore`. Do not commit private or public key files to the repository.

This directory holds the **ECDSA cryptographic key pairs** required for SuperNode identity verification when SuperLink runs with `--enable-supernode-auth`.

---

## 🛠️ Authentication Workflow

SuperNode authentication operates on an asymmetric public/private key model:

1. **Private Keys (`supernode_X`)**: Mounted into individual `supernode` containers. Used to sign gRPC challenge requests sent to SuperLink.
2. **Public Keys (`supernode_X.pub`)**: Registered on SuperLink using `flwr supernode register`. SuperLink uses these to verify SuperNode signatures before granting access to run `ClientApp` execution tasks.

---

## 📁 Expected Directory Structure

After running the generation loop, this folder will contain the following key pairs:

| File Name         | Description                 | Mounted / Used By       |
| :---------------- | :-------------------------- | :---------------------- |
| `supernode_1`     | Private Key for SuperNode 1 | `supernode-1` container |
| `supernode_1.pub` | Public Key for SuperNode 1  | Registered on SuperLink |
| `supernode_2`     | Private Key for SuperNode 2 | `supernode-2` container |
| `supernode_2.pub` | Public Key for SuperNode 2  | Registered on SuperLink |
| `supernode_3`     | Private Key for SuperNode 3 | `supernode-3` container |
| `supernode_3.pub` | Public Key for SuperNode 3  | Registered on SuperLink |

---

## 🚀 Key Generation & Registration Steps

Run these commands from the repository root when setting up or resetting node authentication keys.

### Step 1: Generate ECDSA 256 Key Pairs

Use a bash loop to generate key pairs for `supernode-1`, `supernode-2`, and `supernode-3`:

```bash
for i in 1 2 3; do
  ssh-keygen -t ecdsa -b 256 -N "" -f keys/supernode_${i}
done
```
