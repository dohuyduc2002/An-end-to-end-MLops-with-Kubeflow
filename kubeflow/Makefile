.PHONY: create-cluster enable-addons get-subnet configure-lb install-knative-serving install-istio integrate-istio-knative \
configure-mtls configure-dns verify-istio-installation install-cert-manager install-kserve install-runtimes install-minio \
add-minio-entry create-model-bucket upload-model add-inference-service-entry infer watch-inference-pods infer-concurrent

CLUSTER_NAME ?= kserve
CLUSTER_CPU ?= 10
CLUSTER_MEMORY ?= 12g
CLUSTER_DISK_SIZE ?= 40g

KNATIVE_VERSION := v1.13.1
ISTIO_VERSION := 1.20.3
CERT_MANAGER_VERSION := v1.14.3
KSERVE_VERSION := v0.11.0

create-cluster:
	minikube -p ${CLUSTER_NAME} start --cpus=${CLUSTER_CPU} --memory=${CLUSTER_MEMORY} --disk-size=${CLUSTER_DISK_SIZE}

enable-addons:
	minikube -p ${CLUSTER_NAME} addons enable ingress
	minikube -p ${CLUSTER_NAME} addons enable ingress-dns
	minikube -p ${CLUSTER_NAME} addons enable metallb

get-subnet:
	sudo CLUSTER_NAME=${CLUSTER_NAME} scripts/get-subnet.sh

configure-lb:
	minikube -p ${CLUSTER_NAME} addons configure metallb 

install-knative-serving:
	URL=https://github.com/knative/serving/releases/download/knative-${KNATIVE_VERSION}/serving-crds.yaml \
	FILE=serving-crds.yaml TARGET_DIR=deploy/knative-serving scripts/get-and-apply.sh 
	URL=https://github.com/knative/serving/releases/download/knative-${KNATIVE_VERSION}/serving-core.yaml \
	FILE=serving-core.yaml TARGET_DIR=deploy/knative-serving scripts/get-and-apply.sh 

install-istio:
	curl -L https://istio.io/downloadIstio | ISTIO_VERSION=${ISTIO_VERSION} TARGET_ARCH=x86_64 sh -
	istio-${ISTIO_VERSION}/bin/istioctl manifest generate > deploy/istio/generated-manifest.yaml
	istio-${ISTIO_VERSION}/bin/istioctl install -y

integrate-istio-knative:
	URL=https://github.com/knative/net-istio/releases/download/knative-${KNATIVE_VERSION}/net-istio.yaml \
	FILE=net-istio.yaml TARGET_DIR=deploy/istio scripts/get-and-apply.sh

configure-mtls:
	kubectl label namespace knative-serving istio-injection=enabled
	kubectl apply -f deploy/istio/peer-authentication.yaml

configure-dns:
	kubectl patch configmap/config-domain \
  	--namespace knative-serving \
  	--type merge \
  	--patch '{"data":{"${CLUSTER_NAME}.minikube":""}}'

verify-istio-installation:
	istio-${ISTIO_VERSION}/bin/istioctl verify-install

install-cert-manager:
	URL=https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml \
	FILE=cert-manager.yaml TARGET_DIR=deploy/cert-manager scripts/get-and-apply.sh

install-kserve:
	URL=https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve.yaml \
	FILE=kserve.yaml TARGET_DIR=deploy/kserve scripts/get-and-apply.sh

install-runtimes:
	URL=https://github.com/kserve/kserve/releases/download/${KSERVE_VERSION}/kserve-runtimes.yaml \
	FILE=kserve-runtimes.yaml TARGET_DIR=deploy/kserve scripts/get-and-apply.sh

install-minio:
	kubectl apply -k deploy/minio

add-minio-entry:
	kubectl -n model-storage get svc minio -o json | jq -r .status.loadBalancer.ingress[0].ip > minio_ip
	sudo CLUSTER_NAME=${CLUSTER_NAME} scripts/add-minio-entry.sh
	rm minio_ip

create-model-bucket:
	mc config host add model-storage http://minio.${CLUSTER_NAME}.minikube:9000 minio minio123
	mc mb model-storage/models/sklearn/iris/1.0/models

upload-model:
	mc cp models/sklearn/iris/1.0/models/model.joblib model-storage/models/sklearn/iris/1.0/models/model.joblib

add-inference-service-entry:
	kubectl -n istio-system get svc istio-ingressgateway -o json | jq -r .status.loadBalancer.ingress[0].ip > sklearn_s3_ip
	sudo CLUSTER_NAME=${CLUSTER_NAME} scripts/add-inference-service-entry.sh
	rm sklearn_s3_ip

infer:
	curl -X POST http://sklearn-iris-predictor.examples.${CLUSTER_NAME}.minikube/v2/models/sklearn-iris/infer \
    -H 'accept: application/json' -H 'Content-Type: application/json' \
    --data @data/iris-input.json | jq

watch-inference-pods:
	kubectl -n examples get po -w

infer-concurrent:
	./hey -z 30s -c 300 -m POST -D data/iris-input.json http://sklearn-iris-predictor.examples.${CLUSTER_NAME}.minikube/v2/models/sklearn-iris/infer
