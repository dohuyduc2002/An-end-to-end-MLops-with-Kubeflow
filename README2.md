Install Clickhouse with operator
```bash
k config set-context --current --namespace=database && bash ./k8s/clickhouse/install.sh
# After ClickHouse Operator is installed, you can install ClickHouse cluster with this command
k apply -f ./k8s/clickhouse/clickhouse.yaml -n database
```


```bash
helm upgrade --install dbeaver ./helm-charts/dbeaver -n database 
```

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

helm upgrade --install kafka-infra ./helm-charts/kafka/kafka-infra -n kafka

helm upgrade --install kafka-connect ./helm-charts/kafka/kafka-connect -n kafka

k apply -f k8s/postgres/

k apply -f ./k8s/jobs/insert_application.yaml

k apply -f ./k8s/jobs/insert_bureau.yaml