declare -A vs_map=(
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
HOST="kubeflow.${IP}.nip.io"
echo "🌐 External Host detected: $HOST"

# 1. Patch Gateway
echo "🔧 Patching kubeflow-gateway to accept $HOST ..."
kubectl patch gateway kubeflow-gateway -n kubeflow \
  --type='json' \
  -p="[ { \"op\": \"add\", \"path\": \"/spec/servers/0/hosts/-\", \"value\": \"$HOST\" } ]"

# 2. Patch all VirtualService
for vs in "${!vs_map[@]}"; do
  ns="${vs%%/*}"
  name="${vs##*/}"
  echo "🔧 Patching $vs ..."
  kubectl patch virtualservice "$name" -n "$ns" \
    --type='json' \
    -p="[ { \"op\": \"add\", \"path\": \"/spec/hosts/-\", \"value\": \"$HOST\" } ]"
done
