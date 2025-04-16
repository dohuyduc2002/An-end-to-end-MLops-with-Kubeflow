# ML Platform on Kubernetes: How-To Guide

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Kubernetes Cluster Setup](#kubernetes-cluster-setup)
- [Kubeflow Installation](#kubeflow-installation)
- [Monitoring with Prometheus and Grafana](#monitoring-with-prometheus-and-grafana)
- [CI/CD with Jenkins](#cicd-with-jenkins)
- [MLflow for Experiment Tracking](#mlflow-for-experiment-tracking)
- [Data Management with MinIO and DVC](#data-management-with-minio-and-dvc)
- [Using the UnderWritingModel](#using-the-underwritingmodel)
- [Troubleshooting](#troubleshooting)

## Overview

This guide provides instructions for setting up and using a comprehensive Machine Learning platform built on Kubernetes. The platform integrates Kubeflow, MLflow, monitoring services, CI/CD pipelines, and data versioning tools to create a complete MLOps environment.

## Prerequisites

Before starting, ensure you have:

- A Kubernetes cluster (local or cloud-based)
- kubectl installed and configured
- Helm package manager
- Python 3.8+ and pip
- Docker
- Git

## Kubernetes Cluster Setup

If using Minikube for local development:

```bash
# Start Minikube with sufficient resources
minikube start --memory=8192 --cpus=4 --disk-size=30g
```

For cloud providers (GCP, AWS, Azure), refer to their documentation for cluster setup.

## Kubeflow Installation

### Installation Steps

1. Clone the Kubeflow manifests repository:
   ```bash
   git clone https://github.com/kubeflow/manifests.git
   cd manifests
   ```

2. Install Kubeflow:
   ```bash
   while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
   ```

### Accessing Kubeflow UI

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

Access the Kubeflow dashboard at http://localhost:8080

## Monitoring with Prometheus and Grafana

### Install Prometheus Stack

```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack
helm install kps prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

### Access Grafana

```bash
# Get Grafana admin password
kubectl --namespace monitoring get secrets kps-grafana -o jsonpath="{.data.admin-password}" | base64 -d

# Port-forward to access Grafana (default credentials: admin / prom-operator)
kubectl --namespace monitoring port-forward $POD_NAME 3000
```

Access Grafana at http://localhost:3000

### Access Prometheus

```bash
# Port-forward to access Prometheus
kubectl port-forward -n monitoring prometheus-kps-kube-prometheus-stack-prometheus-0 9090
```

Access Prometheus at http://localhost:9090

## CI/CD with Jenkins

### Install Jenkins

```bash
# Add Jenkins Helm repository
helm repo add jenkins https://charts.jenkins.io
helm repo update

# Install Jenkins
helm install cicd jenkins/jenkins \
  --namespace cicd \
  --create-namespace \
  --set controller.servicePort=6060 \
  --set controller.targetPort=6060
```

### Access Jenkins

```bash
# Get admin password
kubectl get secret --namespace cicd cicd-jenkins -o jsonpath="{.data.jenkins-admin-password}" | base64 --decode

# Port-forward to access Jenkins UI
kubectl port-forward svc/cicd-jenkins 6060:6060 -n cicd
```

Access Jenkins at http://localhost:6060 (Username: admin, Password: retrieved from command)

### Remove Jenkins

```bash
helm uninstall cicd --namespace cicd
kubectl delete namespace cicd
```

## MLflow for Experiment Tracking

### Install MLflow

```bash
# Add MLflow Helm repository
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update

# Install MLflow
helm install mlflow community-charts/mlflow \
  --namespace mlflow \
  --create-namespace
```

### Access MLflow UI

```bash
# Port-forward to access MLflow
kubectl port-forward svc/mlflow -n mlflow 5003:5000
```

Access MLflow at http://localhost:5003

### Restart or Remove MLflow

```bash
# Restart MLflow pods
kubectl delete pods -n mlflow --selector=app.kubernetes.io/name=mlflow

# Uninstall MLflow
helm uninstall mlflow -n mlflow
```

## Data Management with MinIO and DVC

### Access MinIO

```bash
# Port-forward to access MinIO
kubectl -n kubeflow port-forward pod/minio-6748f5ff9d-bx6v5 9000:9000
```

### Configure MinIO Client

```bash
# Configure MinIO client
mc config host add localMinio http://localhost:9000/minio minio minio123

# List buckets
mc ls localMinio

# Upload data to MinIO
mc cp --recursive data localMinio/sample-data/
mc ls --recursive localMinio/sample-data
```

### Set Up DVC with MinIO

```bash
# Initialize DVC in your project
dvc init

# Add MinIO as remote storage
dvc remote add -d myminio s3://sample-data/data
dvc remote modify myminio endpointurl http://localhost:9000
dvc remote modify myminio access_key_id minio
dvc remote modify myminio secret_access_key minio123

# Start tracking data
dvc add data/
git add data.dvc .gitignore
git commit -m "Add data tracking with DVC"

# Push data to remote storage
dvc push
```

## Using the UnderWritingModel

### Installation

```bash
git clone <repository-url>
cd under_writing
pip install -r requirements.txt
```

### Usage Steps

1. **Data Preparation**:
   ```python
   from utils import preprocess_data, handle_missing_values, detect_outliers
   
   # Preprocess your data
   processed_data = preprocess_data(raw_data)
   processed_data = handle_missing_values(processed_data)
   processed_data, outliers = detect_outliers(processed_data)
   ```

2. **Model Training**:
   ```python
   from UnderWritingModel import UnderWritingModel
   
   # Initialize and train the model
   model = UnderWritingModel(X_train, y_train)
   model.train_model(model_type="xgb")  # or "lgbm" for LightGBM
   
   # Make predictions
   predictions = model.model.predict(X_test)
   ```

3. **Model Evaluation**:
   ```python
   # Log model with MLflow
   model.log_model(model_type="xgb", version="v1")
   ```

4. **Model Interpretation**:
   ```python
   # Get SHAP values for model explanation
   shap_values = model.explain_model(data="test")
   ```

### Example with UnderWritingTrainer

```python
from src.underwriting_trainer import UnderWritingTrainer

processed_train = "data/processed/train/processed_train_v1.csv"
processed_test = "data/processed/test/processed_test_v1.csv"

trained_model = UnderWritingTrainer.train_model(
    model_name="xgb",
    processed_train=processed_train,
    processed_test=processed_test,
    version="v1",
    experiment_name="UnderWriting_Experiment"
)
```

## Troubleshooting

### Common Issues

#### Kubernetes Resources Management

```bash
# List all pods in a namespace
kubectl get pods -n <namespace>

# Get pod logs
kubectl logs -n <namespace> <pod-name>

# List and manage persistent volume claims
kubectl get pvc -n kubeflow-user-example-com
kubectl delete pvc mlops-test-workspace -n kubeflow-user-example-com
```

#### MinIO Connection Issues

If you can't connect to MinIO, check the pod status:

```bash
kubectl get pods -n kubeflow | grep minio
```

#### MLflow Connection Issues

If MLflow isn't accessible, check pod logs:

```bash
kubectl get pods -n mlflow
kubectl logs -n mlflow <mlflow-pod-name>
```

#### Kubeflow UI Access Issues

If you can't access the Kubeflow UI, verify the istio-ingressgateway service:

```bash
kubectl get svc -n istio-system
```

### Miscellaneous Commands

```bash
# Get nodes information
kubectl get nodes

# Get all services across namespaces
kubectl get svc --all-namespaces

# Get all deployments
kubectl get deployments --all-namespaces

# Describe a specific resource
kubectl describe pod <pod-name> -n <namespace>
```

For more detailed troubleshooting, refer to the documentation of each component.