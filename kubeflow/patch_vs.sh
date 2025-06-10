# First we declare an array named virtial_service_map that maps the namespace and name of each VirtualService to a more readable name.
declare -A virtial_service_map=(
  [auth/dex]="dex"
  [kubeflow/centraldashboard]="centraldashboard"
  [kubeflow/jupyter-web-app-jupyter-web-app]="jupyter"
  [kubeflow/katib-ui]="katib"
  [kubeflow/kserve-models-web-app]="kserve"
  [kubeflow/metadata-grpc]="metadata"
  [kubeflow/ml-pipeline-ui]="mlpipeline"
  [kubeflow/profiles-kfam]="profiles"
  [kubeflow/tensorboards-web-app-tensorboards-web-app]="tensorboards"
  [kubeflow/volumes-web-app-volumes-web-app]="volumes"
  [oauth2-proxy/oauth2-proxy]="oauth2"
)

# 0. Auto detect LoadBalancer IP
IP=$(kubectl get svc istio-ingressgateway-lb -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
HOST="kubeflow.ducdh.com"
echo "🌐 External IP detected: $IP"
echo "🌐 Patching Host : $HOST"

# 1. Patch Gateway
# refer to this https://github.com/kubeflow/manifests/blob/v1.10-branch/common/istio-1-24/cluster-local-gateway/base/gateway.yaml
# refer to this https://datatracker.ietf.org/doc/html/rfc6902#section-4.1
# We use the json default patch to add the host to the kubeflow-gateway
# In this case \"op\": \"add\", \"path\" this will point the operator to add with specific path in yaml
# \"/spec/servers/0/hosts/-\", \"value\": \"$HOST\ : this will be patched to the kubeflow-gateway spec with value of $HOST
echo "🔧 Patching kubeflow-gateway to accept $HOST ..."
kubectl patch gateway kubeflow-gateway -n kubeflow \
  --type='json' \
  -p="[ { \"op\": \"add\", \"path\": \"/spec/servers/0/hosts/-\", \"value\": \"$HOST\" } ]"

# 2. Patch all VirtualService
# Iterate through all keys in associative array 'virtial_service_map'.
# All key names are formatted as "namespace/vs_name" (ví dụ: "kubeflow/centraldashboard").
# All virtualservices yaml are under apps in kubeflow manifests repo 
# In deployment, all virtualservices are created in the kubeflow namespace.
# refer to this sample https://github.com/kubeflow/manifests/blob/v1.10-branch/apps/jupyter/jupyter-web-app/upstream/overlays/istio/virtual-service.yaml
for vs in "${!virtial_service_map[@]}"; do
  # Get namespace, remove the prefix after namespace/.... "/*".
  ns="${vs%%/*}"

  # Get the name of the VirtualService, which is the part after the last '/' in the key.
  name="${vs##*/}"
  echo "🔧 Patching $vs ..."
  kubectl patch virtualservice "$name" -n "$ns" \
    --type='json' \
    -p="[ { \"op\": \"add\", \"path\": \"/spec/hosts/-\", \"value\": \"$HOST\" } ]"
done
