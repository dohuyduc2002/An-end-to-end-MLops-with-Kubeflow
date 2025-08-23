helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer \
  --set controller.service.externalTrafficPolicy=Cluster \
  --set controller.resources.requests.cpu=100m \
  --set controller.resources.requests.memory=90Mi \
  --set controller.config.proxy-body-size="5120G" \
  --set controller.config.proxy-connect-timeout="600" \
  --set controller.config.proxy-send-timeout="600" \
  --set controller.config.proxy-read-timeout="600"

------------
docker login

kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=$HOME/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson

while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 20; done

k create secret generic minio-creds \
  --from-literal=access_key=minio \
  --from-literal=secret_key=minio123 \
  -n kubeflow
------------
helm install minio minio/minio \
  --namespace minio \
  --create-namespace \
  --set mode=standalone \
  --set rootUser=minio \
  --set rootPassword=minio123 \
  --set persistence.size=10Gi \
  --set service.type=ClusterIP \
  --set resources.requests.memory=1Gi \
  --set ingress.enabled=true \
  --set ingress.ingressClassName=nginx \
  --set ingress.hosts[0]=minio.ducdh.com \
  --set consoleIngress.enabled=true \
  --set consoleIngress.ingressClassName=nginx \
  --set consoleIngress.hosts[0]=console.minio.ducdh.com 
------------
mc alias set localMinio http://minio.ducdh.com minio minio123
mc mb localMinio/feast
mc mb localMinio/mlflow
mc mb localMinio/sample-data
mc mb localMinio/stream-bucket
mc mb localMinio/flink-data
mc mb localMinio/gold
mc mb localMinio/silver

mc cp --recursive ./data/ localMinio/sample-data
mc ls --recursive localMinio/sample-data

------------

k create ns database

k apply -f k8s/postgres/

helm install mlflow community-charts/mlflow \
  --namespace mlflow \
  --create-namespace \
  -f helm-charts/mlflow/custom-values.yaml


------------
```bash
bash ./k8s/clickhouse/install.sh
# After ClickHouse Operator is installed, you can install ClickHouse cluster with this command
k apply -f ./k8s/clickhouse/clickhouse.yaml -n database
```


```bash
helm upgrade --install dbeaver ./helm-charts/dbeaver -n database 
```
------------
helm install strimzi strimzi/strimzi-kafka-operator \
  --version 0.46.0 \
  --create-namespace \
  --namespace kafka

helm upgrade --install kafka-infra ./helm-charts/kafka/kafka-infra -n kafka

helm upgrade --install kafka-connect ./helm-charts/kafka/kafka-connect -n kafka
----------------

k apply -f k8s/postgres/

k apply -f ./k8s/jobs/insert_application.yaml

k apply -f ./k8s/jobs/insert_bureau.yaml
k apply -f ./k8s/jobs/insert_bureau_balance.yaml
----------------
helm install flink-kubernetes-operator flink-operator-repo/flink-kubernetes-operator \
  -n flink \
  --create-namespace

k create secret generic minio-creds \
  --from-literal=access_key=minio \
  --from-literal=secret_key=minio123 \
  -n flink

k apply -f ./k8s/flink/flink_deployment.yaml -n flink

----------------

kubectl create namespace feast-operator-system
kubectl apply -f https://raw.githubusercontent.com/feast-dev/feast/refs/tags/v0.49.0/infra/feast-operator/dist/install.yaml --namespace=feast-operator-system

helm upgrade --install feast ./helm-charts/feast/ \
  --namespace feast-operator-system