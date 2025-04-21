# README.md is under implementation

* Contents:
  * [Introduction](#introduction)
	* [Repository structure](#repository-structure)
	* [Prerequisites installation](#prerequisites-installation)
	* [Component Preparation](#component-preparation)
	* [Usage](#usage)
	* [Additional Usage](#additional-usage)
<!-- /code_chunk_output -->

## Introduction
This project is an unified platform for Datasciene team whom working on Credit modeling sector. This repo will help and guide you to build and serve ML model as in a production environment (Google Cloud Platform). I also used tool & technologies to quickly deploy the ML system into production and automate processes during the development and deployment of the ML system.

## Repository structure
```txt
Underwriting prediction
├── data.dvc                              
├── HOW-TO-GUIDE.md
├── ingress                                                  * Ingress controller for all services 
│   └── ingress-underwrite.yaml                       
├── jenkins                                                  * Jenkins community helm chart
│   ├── charts
│   ├── CONTRIBUTING.md
│   ├── ct.yaml
│   ├── LICENSE
│   ├── PROCESSES.md
│   └── README.md
├── kubeflow                                                  * Kubeflow manifest 1.10
│   ├── a
│   ├── kubeflow-cluster.yaml
│   ├── manifests
│   ├── namespace
│   └── README.md
├── LICENSE
├── mlflow                                                    * MLflow community helm chart
│   ├── Chart.lock
│   ├── Chart.yaml
│   ├── files
│   ├── LICENSE
│   ├── mlflow-network-policy.yaml
│   ├── postgres-mlflow.yaml
│   ├── README.md
│   ├── README.md.gotmpl
│   ├── templates
│   ├── unittests
│   ├── values-kind.yaml
│   ├── values.schema.json
│   └── values.yaml
├── monitor                                                * Kube-prometheus-stack helm chart         
│   ├── charts
│   ├── Chart.yaml
│   ├── ci
│   ├── CONTRIBUTING.md
│   ├── files
│   ├── hack
│   ├── README.md
│   ├── templates
│   ├── unittests
│   ├── UPGRADE.md
│   └── values.yaml
├── README.md
├── src                                                 * Source code for the project           
│   ├── kfp                                             * Kubeflow pipeline
│   ├── observability                                   * Observability and API endpoint           
│   ├── pipeline_deprecated                             * Deprecated Kubeflow pipeline
│   └── __pycache__
└── test                                                * Test files for the project         
    ├── jenkins
    ├── test_minio_dvc.sh
    └── track_minio_data.sh
```
## To-Do
- [x] Add pkl joblib transform process into pipeline and app
- [x] Add support for other models (e.g., LightGBM, CatBoost)
- [x] Create automated pipeline for model training and evaluation
- [ ] Fix webhook arlert to Discord
- [ ] Using Ingress controller for all services 
- [ ] Implement more metrics monitoring with Evidently
- [ ] Implement Unit test for all functions
- [ ] Add CI/CD pipeline using Jenkins
- [ ] Insantiate Terraform for IaC to deploy in GCP k8s
- [ ] Move data into GCP and using DVC for versioning
- [ ] Implement Data Ingestion, Data Quality check, Data Lake, Data Warehouse, and Data Pipeline


## Prerequisites installation
This project is runinning on Kind cluster, **Kind** (Kubernetes IN Docker) is a tool for running local Kubernetes clusters using Docker container "nodes." It is primarily designed for testing Kubernetes itself, CI pipelines, and local development environments. Kind is lightweight, easy to set up, and supports multi-node cluster configurations.

This is the environment I used to run this project:
- Client Version: v1.32.3
- Kustomize Version: v5.5.0
- Server Version: v1.32.0
- Kind v0.27.0 (go1.20.5)

### Install Golang

Since both Kubernetes and Kind are written in Golang, you need to install Golang first.  
You can follow the official [Golang installation guide](https://golang.org/doc/install) or run the following commands:

```bash
sudo apt update
sudo apt install -y golang-go
```

### Install Kind
You can install Kind by running the following command:

```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.27.0/kind-$(uname)-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

Be sure to check Kustomize version cause this will be used to deploy Kubeflow. 

```
curl -Lo kustomize.tar.gz https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.5.0/kustomize_v5.5.0_linux_amd64.tar.gz
tar -xzf kustomize.tar.gz
chmod +x kustomize
sudo mv kustomize /usr/local/bin/
```

Install Krew for Kubectl plugins, you can install Krew by following this link: [Krew installation](https://krew.sigs.k8s.io/docs/user-guide/setup/install/)


For convinience when using Kubeflow, you can install these Kubectl plugins and alias:
```
echo "alias k=kubectl" >> ~/.bashrc
source ~/.bashrc
kubectl krew install ctx
kubectl krew install ns
echo "alias kubectx='kubectl ctx'" >> ~/.bashrc
echo "alias kubens='kubectl ns'" >> ~/.bashrc
```

That all the prerequisites you need to install. 

## Component Preparation
In this section, I will guide you to install and configure all the components in this project.
### Initialize Kubeflow cluster
Kubeflow is an open-source platform designed to facilitate the deployment, orchestration, and management of machine learning (ML) workflows on Kubernetes. It provides a set of tools and components that enable data scientists and ML engineers to build, train, and deploy ML models at scale.

To install Kubeflow, first you clone the Kubeflow manifest repo [Kubeflow manifest 1.10](https://github.com/kubeflow/manifests/tree/v1.10-branch). I have already cloned this repo in `kubeflow/manifests` folder. 

After that, you can install Kubeflow using the README file in `kubeflow/manifests` folder. 

### Initialize Mlflow 
MLflow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models.

Im using MLflow community helm chart to deploy MLflow in this project. You can find the helm chart in `mlflow` folder which is cloned from this repo [MLflow community helm chart](https://github.com/community-charts/helm-charts/tree/main/charts/mlflow)

I'm using Postgres as backend store and Minio as artifact store. This can be configure using this cmd

```bash
k apply -f postgres-mlflow.yaml
```
After initialize MLflow, we init Minio as artifact store

```bash
helm upgrade --install mlflow community-charts/mlflow \
  --namespace mlflow \
  --create-namespace \
  \
  --set backendStore.databaseMigration=true \
  --set backendStore.postgres.enabled=true \
  --set backendStore.postgres.host=postgres-service \
  --set backendStore.postgres.port=5432 \
  --set backendStore.postgres.database=postgres \
  --set backendStore.postgres.user=postgres \
  --set backendStore.postgres.password=postgres \
  \
  --set artifactRoot.s3.enabled=true \
  --set artifactRoot.s3.bucket=mlflow \
  --set artifactRoot.s3.awsAccessKeyId=minio \
  --set artifactRoot.s3.awsSecretAccessKey=minio123 \
  \
  --set extraEnvVars.AWS_ACCESS_KEY_ID=minio \
  --set extraEnvVars.AWS_SECRET_ACCESS_KEY=minio123 \
  --set extraEnvVars.AWS_REGION=us-east-1 \
  --set extraEnvVars.MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  --set extraEnvVars.MLFLOW_S3_IGNORE_TLS="true" \
  --set extraEnvVars.AWS_S3_ADDRESSING_STYLE="path" \
  \
  --set serviceMonitor.enabled=true
```

Be cause Minio is in `mlflow` namespace, we need to apply network policy to allow MLflow to access Minio. 

```bash
kubectl apply -f mlflow-network-policy.yaml
```

### Initialize Jenkins 
Jenkins is an open-source automation server that enables developers to build, test, and deploy their software. It provides hundreds of plugins to support building, deploying, and automating any project.

In this repo, I'm using Jenkins community helm chart to deploy Jenkins in this project. You can find the helm chart in `jenkins` folder which is cloned from this repo [Jenkins community helm chart](https://github.com/jenkinsci/helm-charts)

```bash
helm install cicd jenkins/jenkins \
  --namespace cicd \
  --create-namespace \
  --set controller.servicePort=6060 \
  --set controller.targetPort=6060
  ```

### Initialize Prometheus-Grafana
To monitor the system, I'm using Prometheus and Grafana. Prometheus is an open-source systems monitoring and alerting toolkit originally built at SoundCloud. Grafana is an open-source platform for monitoring and observability. I'm using Kube-prometheus-stack helm chart to deploy Prometheus and Grafana in this project. You can find the helm chart in `monitor` folder which is cloned from this repo [Kube-prometheus-stack helm chart](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
#### Prometheus


#### Grafana

Create json for Grafana dashboard, apply it through configmap

kubectl create configmap model-gini-dashboard \
  --from-file=model-gini-dashboard.json \
  -n monitoring \
  -o yaml --dry-run=client | kubectl apply -f -


### Ingress controller for all services

## Usage 

### Using Kubeflow 
For simplicity, in this project I used default Kubeflow namespace which is `kubeflow-user-example-com`. You can create your own namespace by using the following command:
```bash
kubectl create namespace <your-namespace>
```
Then, you can use the following command to create a new Kubeflow notebook:
----

### Using Kubeflow Notebook 
#### Create Notebook workspace 
Under this `kubeflow-user-example-com` namespace, we create new notebook pod with following command from the UI 

video....

1. Initialize git inside the notebook pod
You can consider this as a local environment, so you can use the following command to initialize git inside the notebook pod:

video... 


This repo is pushed inside Kubeflow notebook: [git-underwrite-mlflow](https://github.com/dohuyduc2002/git-underwrite-mlflow) 


### Using Kserve

under implementation

### Using Kubeflow Pipeline


### Serve model with FastAPI and collect log 

### CICD with Jenkins

### Monitoring with Grafana, Prometheus and Evidently


# K8s - grafana - prometheus
kubectl --namespace monitoring get secrets kps-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo 
kubectl --namespace monitoring port-forward kps-grafana-6f79cb6d98-ph54q 3000
(admin - pwd: prom-operator)

helm install kps prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# jenkins helm
helm install cicd jenkins/jenkins \
  --namespace cicd \
  --create-namespace \
  --set controller.servicePort=6060 \
  --set controller.targetPort=6060

helm uninstall cicd --namespace cicd
kubectl delete namespace cicd

# get username, pass
kubectl get secret kps-grafana -n monitoring -o jsonpath="{.data.admin-user}" | base64 --decode && echo
kubectl get secret kps-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 --decode && echo

kubectl port-forward svc/cicd-jenkins 6060:6060 -n cicd

# mlflow
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update




cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: minio
  namespace: mlflow
spec:
  ports:
    - port: 9000
---
apiVersion: v1
kind: Endpoints
metadata:
  name: minio
  namespace: mlflow
subsets:
  - addresses:
      - ip: 10.96.108.10
    ports:
      - port: 9000
EOF



# upload data to mino and add dvc 
mc config host add localMinio http://localhost:9000/minio minio minio123
mc ls localMinio
mc cp --recursive data localMinio/sample-data/
mc ls --recursive localMinio/sample-data

dvc init
dvc remote add -d myminio s3://sample-data/data
dvc remote modify myminio endpointurl http://localhost:9000
dvc remote modify myminio access_key_id minio
dvc remote modify myminio secret_access_key minio123

# delete pvc workspace
kubectl get pvc -n kubeflow-user-example-com

kubectl delete pvc a-workspace -n kubeflow-user-example-com
kubectl port-forward svc/minio-service -n kubeflow 9000:9000


#app.py
uvicorn api:app --host 0.0.0.0 --port 8000
ngrok http --url pheasant-crack-curiously.ngrok-free.app 8000


# monitoring svc 
kubectl create secret generic minio-creds \
  --from-literal=access_key=minio \
  --from-literal=secret_key=minio123 \
  --namespace=monitoring


kubectl label svc prediction-api app=prediction-api -n monitoring --overwrite
