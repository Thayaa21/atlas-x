# ATLAS-X GitOps Deployment (Feature 8)

## Overview

ATLAS-X uses a GitOps workflow powered by ArgoCD.  Every merge to `main`
automatically deploys to the `fraud-detection` Kubernetes namespace.

```
Developer → Git Push → GitHub → ArgoCD watches → kubectl apply → Cluster
```

## Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or local k3s/minikube)
- ArgoCD installed: `kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml`
- `kubectl` and `argocd` CLI configured

## Directory Structure

```
deployment/
├── argocd-app.yaml          # ArgoCD Application manifest
├── README.md                # This file
└── k8s/
    ├── api-deployment.yaml  # FastAPI + HPA
    ├── kafka-deployment.yaml
    ├── neo4j-statefulset.yaml
    ├── postgres-statefulset.yaml  # Uses pgvector/pgvector:pg16 image
    ├── ingress.yaml
    ├── configmap.yaml
    └── secrets.yaml         # TEMPLATE – use Sealed Secrets in production
```

## Initial Setup

### 1. Register the application with ArgoCD

```bash
# Login to ArgoCD
argocd login <argocd-server>

# Create the application
kubectl apply -f deployment/argocd-app.yaml

# Trigger initial sync
argocd app sync atlas-x
```

### 2. Verify deployment

```bash
argocd app get atlas-x
kubectl get pods -n fraud-detection
kubectl get svc  -n fraud-detection
```

### 3. Check API health

```bash
kubectl port-forward svc/atlas-x-api 8000:80 -n fraud-detection
curl http://localhost:8000/api/v1/health
```

## GitOps Workflow

### Normal deployment (feature branch → main)

```
1. Developer creates feature branch
2. Opens PR → CI runs tests
3. PR merged to main
4. ArgoCD detects change (polls every 3 minutes or via webhook)
5. ArgoCD syncs: kubectl apply -f deployment/k8s/
6. Rolling update: zero-downtime (maxUnavailable=0)
7. Readiness probe gates traffic until new pods are healthy
```

### Rollback procedure

```bash
# Option 1: ArgoCD UI – click "History" → select previous version → "Rollback"

# Option 2: CLI rollback to previous revision
argocd app rollback atlas-x <revision-id>

# Option 3: Git revert (preferred – keeps Git as source of truth)
git revert <bad-commit-sha>
git push origin main
# ArgoCD auto-syncs the revert
```

### Emergency rollback (immediate)

```bash
# Rollback the Kubernetes deployment directly (bypasses ArgoCD temporarily)
kubectl rollout undo deployment/atlas-x-api -n fraud-detection

# Then fix Git and let ArgoCD re-sync to restore GitOps state
```

## Secrets Management

**Never commit real secrets to Git.**

Options:
1. **Sealed Secrets** (recommended for self-hosted):
   ```bash
   kubeseal --format yaml < deployment/k8s/secrets.yaml > deployment/k8s/sealed-secrets.yaml
   git add deployment/k8s/sealed-secrets.yaml
   ```

2. **AWS Secrets Manager + External Secrets Operator**:
   ```yaml
   apiVersion: external-secrets.io/v1beta1
   kind: ExternalSecret
   metadata:
     name: atlas-x-secrets
   spec:
     secretStoreRef:
       name: aws-secrets-manager
       kind: ClusterSecretStore
     target:
       name: atlas-x-secrets
     data:
       - secretKey: OPENAI_API_KEY
         remoteRef:
           key: atlas-x/openai-api-key
   ```

3. **HashiCorp Vault** with the Vault Agent Injector.

## Scaling

The API deployment includes an HPA that scales 2–10 replicas based on CPU/memory:

```bash
# Check HPA status
kubectl get hpa -n fraud-detection

# Manual scale (temporary – ArgoCD will revert to HPA-managed count)
kubectl scale deployment atlas-x-api --replicas=5 -n fraud-detection
```

## Model Updates

To deploy a new model version:

1. Train and save: `src/models/atlass_x_xgb_v5.pkl`
2. Update `MODEL_PATH` in `src/api/model_loader.py`
3. Build and push new Docker image: `docker build -t atlas-x-api:v5 .`
4. Update image tag in `deployment/k8s/api-deployment.yaml`
5. Commit and push → ArgoCD auto-deploys

## Monitoring in Kubernetes

```bash
# View API logs
kubectl logs -l app=atlas-x,component=api -n fraud-detection --tail=100 -f

# Port-forward Grafana
kubectl port-forward svc/grafana 3000:3000 -n fraud-detection

# Port-forward Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n fraud-detection
```
