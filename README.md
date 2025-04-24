# An unified platform for Credit modeling Version 1

* Contents:
  * [Introduction](#introduction)
	* [Repository structure](#repository-structure)
	* [Prerequisites installation](#prerequisites-installation)
	* [Component Preparation](#component-preparation)
	* [Usage](#usage)
	* [Additional Usage](#additional-usage)
<!-- /code_chunk_output -->

**Disclaimer**: This is a version 1 of this project, I will keep updating this project to make it more complete and useful.

You can refer to this github repo that I pushed in Kubeflow notebook in this link : [git-underwrite-mlflow](https://github.com/dohuyduc2002/git-underwrite-mlflow), there also documentation in here to setup git and basic usage of Kubeflow notebook workspace. 

![Diagram](media/diagram.jpg)

## Introduction
This project is an unified platform for Datasciene team whom working on Credit modeling sector. This repo will help and guide you to build and serve ML model as in a production environment (Google Cloud Platform). I also used tool & technologies to quickly deploy the ML system into production and automate processes during the development and deployment of the ML system.

## Repository structure
```txt
Underwriting prediction
├── data.dvc
├── helm                                                       * community helm chart for mlflow, kube-prometheus-stack
│   ├── mlflow
│   └── monitor
├── ingress                                                    * Ingress controller for all services
│   ├── allow-ingress-to-minio.yaml
│   ├── ingressgateway-allow-kubeflow.yaml
│   ├── ingress-jenkins.yaml
│   ├── ingress-kubeflow.yaml
│   ├── ingress-minio.yaml
│   ├── ingress-mlflow.yaml
│   ├── ingress-monitoring.yaml
│   └── metallb-config.yaml
├── Jenkinsfile
├── kubeflow                                                   * Kubeflow manifest v1.10
│   ├── kubeflow-cluster.yaml
│   ├── manifests
│   ├── namespace
│   └── README.md
├── LICENSE
├── local                                                      * Custom Jenkins
│   ├── docker-compose.yaml
│   ├── jenkins
│   └── jenkins-service.yaml
├── media
│   └── diagram.jpg
├── README.md
└── src                                                      
    ├── client                                                 * Source code for client api endpoint and test
    │   ├── app
    │   ├── data
    │   ├── Dockerfile
    │   ├── download_data.sh
    │   ├── download_joblib.py
    │   ├── grafana
    │   ├── joblib
    │   ├── k8s
    │   ├── monitor
    │   ├── requirements.txt
    │   ├── test
    │   └── test.json
    └── kfp                                                   * Source code to run kubeflow pipeline in local 
```
## To-Do
- [x] Add pkl joblib transform process into pipeline and app
- [x] Add support for other models (e.g., LightGBM, CatBoost)
- [x] Create automated pipeline for model training and evaluation
- [ ] Add logging in new pipeline to show log in test
- [ ] Fix webhook arlert to Discord (Currenly in Firing state)
- [ ] Using Ingress controller for all services 
- [ ] Implement more metrics monitoring with Evidently
- [ ] Refractor test to avoid code duplication
- [x] Implement Unit test for all functions
- [x] Add CI/CD pipeline using Jenkins
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
After initialize MLflow, we bind to Minio as artifact store, before that you have to **forward Minio service port** (this will be implement later)

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

Be cause Minio is in `kubeflow` namespace, we need to apply network policy to allow MLflow to access Minio. 

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
To get user and password for Jenkins, you can run the following command:

```bash
k get secrets -n cicd
```
After that using this command to get password:
```bash
k get secret cicd-jenkins -n cicd -o jsonpath="{.data}"
```

### Initialize Prometheus-Grafana
To monitor the system, I'm using Prometheus and Grafana. Prometheus is an open-source systems monitoring and alerting toolkit originally built at SoundCloud. Grafana is an open-source platform for monitoring and observability. I'm using Kube-prometheus-stack helm chart to deploy Prometheus and Grafana in this project. You can find the helm chart in `monitor` folder which is cloned from this repo [Kube-prometheus-stack helm chart](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
#### Prometheus
I'm setting up Prometheus to monitor system metric through OpenTelemetry. Under the `monitor` namespace, apply these configs in `src/observability/k8s` to deploy Prometheus and Grafana. 

```bash
k apply -f alertmanager-config.yaml
k apply -f minio-allow-monitoring.yaml
k apply -f underwriting-alerts.yaml
```
Due to Mino is in `kubeflow` namespace, we need to apply network policy to allow Prometheus to access Minio. 

#### Grafana
Grafana is a powerful open-source analytics and monitoring solution that integrates with various data sources, including Prometheus. It provides a rich set of features for visualizing and analyzing time-series data.

Create json for Grafana dashboard, apply it through configmap in `src/client/grafana` folder

```bash
k create configmap model-gini-dashboard \
  --from-file=model-gini-dashboard.json \
  -n monitoring \
  -o yaml --dry-run=client | kubectl apply -f -
```

### Ingress controller for all services
Only ingressed services in Kind cluster, can only access in localhost through domain, using path-based routing, cannot expose to other machine. 

- **Limitation** Kind was created *without* `extraPortMappings`; Docker does **not** let you add host-port mappings to running node-containers, so ports 80/443 stay invisible to the outside world. :contentReference[oaicite:0]{index=0}  
- All LoadBalancer IPs that MetalLB assigns live only inside Docker’s bridge network, making Ingress reachable solely from the host machine. :contentReference[oaicite:1]{index=1}  
- Re-creating the Kind cluster is the only “native” way to expose those ports, but you prefer not to delete the current cluster. :contentReference[oaicite:2]{index=2}  

- **Proposed solution** Spin up a fresh **Minikube** cluster instead.  
- Enable the built-in `ingress` and `metallb` add-ons with one command, giving you a real LoadBalancer IP on the host interface. :contentReference[oaicite:3]{index=3}  
- Optionally start `minikube tunnel` to bridge LoadBalancer traffic to the host if you skip MetalLB. :contentReference[oaicite:4]{index=4}  
- Add the `ingress-dns` add-on (or normal DNS A record) so domain names resolve to that IP without editing `/etc/hosts`. :contentReference[oaicite:5]{index=5}  
- With ports 80/443 now exposed at the host’s public address, you can attach a DNS record and obtain TLS certificates as usual.  
- Kubeflow and other services keep their path-based routing rules inside Ingress objects—no manifest changes needed.  
- Result: public, domain-based access to all services, achieved without tearing down your existing Kind environment.

## Usage 

### Using Kubeflow 
For simplicity, in this project I used default Kubeflow namespace which is `kubeflow-user-example-com`. You can create your own namespace by using the following command:
```bash
kubectl create namespace <your-namespace>
```
After that, you can follow tutorial in this git repo [git-underwrite-mlflow](https://github.com/dohuyduc2002/git-underwrite-mlflow) to setup kubeflow workspace from the UI and git. 

### Using Kserve

under implementation, fixing bug in Kserve *v0.14.1*

### Using Kubeflow Pipeline
**Kubeflow Pipelines** is a powerful platform for building and deploying scalable and reproducible machine learning (ML) workflows based on Kubernetes. It allows data scientists and ML engineers to define workflows as a series of components, each performing a specific task (e.g., preprocessing, training, evaluation).

With Kubeflow Pipelines, you can:
- Track experiments and compare results visually.
- Automate the ML lifecycle from data ingestion to model deployment.
- Reuse pipeline components across projects.
- Scale easily using Kubernetes-native resources.

Ideal for teams working on MLOps, Kubeflow Pipelines simplifies the path from prototype to production.

*Disclaimer*: I'm fixing KFP running in Kubeflow notebook workspace *Inside the cluster*, until now, you can run it *Outside the cluster*, which i have configured in `src/kfp` folder.

image pipeline 

To run the pipeline, you should following these steps
1. Create components yaml 
2. Create pipeline yaml
3. Upload pipeline to Kubeflow

### Tracking data with DVC 
In this project, I'm tracking all data under `sample-data` bucket in Minio for simplicity. To track data with DVC, you can follow these steps:
1. Install DVC by running the following command:
```bash
pip install dvc
```
2. Initialize DVC in your project by running the following command:
```bash
mc config host add localMinio http://localhost:9000/minio minio minio123

dvc init
dvc remote add -d myminio s3://sample-data/data
dvc remote modify myminio endpointurl http://localhost:9000
dvc remote modify myminio access_key_id minio
dvc remote modify myminio secret_access_key minio123
```
if you want to push data from local to Minio, you can run the following command:
```bash
mc ls localMinio
mc cp --recursive <yourpath> localMinio/sample-data/
mc ls --recursive localMinio/sample-data
```

3. After that, you can track data by running the following command:
```bash
dvc import-url s3://sample-data/ data/sample-data/ --external
```

4. Commit the changes to DVC by running the following command:
```bash
git add data/sample-data.dvc
git commit -m "......."
```

### Testing CICD with Jenkins

Under fixing bug due to using Jenkins on a VM outside of Kind, if using inside Kind, Jenkins can not find docker daemon. 

### Serve model with FastAPI and collect log 
In the endpoint API, the application is pulling model from Mlflow artifact storage which is under Minio bucket `mlflow`. The model joblib is stored in `mlpieline` bucket. This app consist 2 POST method, one is raw prediction which used to predict new customer which is not in the existed database. The 2nd one is predict by id which customer is already existed in the database. 

I'm also collecting prediction log using OpenTelemetry and send it back to Prometheus. The metrics dashboard is created in Grafana throguh a configmap that created above .

There are 2 ways to deploy endpoint api
1. CICD : The endpoint is automatically deployed when the Jenkins pipeline run success 
2. Manual: You can deploy the endpoint manually by using the following command:

```bash
k port-forward -n monitoring deployment/prediction-api 8000:8000 8001:8001
```

**Note**: You can also use `ngrok` to expose the endpoint to the internet, instruction can be found in this [ngrok](https://ngrok.com/docs/getting-started/installation) page. 


### Monitoring with Grafana, Prometheus and Evidently

To access Prometheus and Grafana, you can use the following command:

```bash
k port-forward -n monitoring deployment/prediction-api 8000:8000 8001:8001
k port-forward -n monitoring prometheus-kps-kube-prometheus-stack-prometheus-0 9090:9090
``` 

