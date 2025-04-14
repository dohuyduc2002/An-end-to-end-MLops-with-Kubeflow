kind delete cluster --name=kubeflow

kind create cluster --name=kubeflow --config=kubeflow-cluster.yaml

kind get kubeconfig --name kubeflow > /tmp/kubeflow-config
export KUBECONFIG=/tmp/kubeflow-config

docker login

kubectl create secret generic regcred \
  --from-file=.dockerconfigjson=/home/s48gb/.docker/config.json \
  --type=kubernetes.io/dockerconfigjson

cert-manager -> isto -> dex -> full

while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 20; done

kustomize edit fix

kustomize build apps/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user | kubectl apply -f -

kubectl -n istio-system describe pod kubeflow-m2m-oidc-configurator-29075670-76qpx
