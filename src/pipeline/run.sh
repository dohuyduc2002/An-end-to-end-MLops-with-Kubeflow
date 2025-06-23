cd scripts
python3 dataloader.py
python3 binning.py
python3 feat_selector.py
python3 modeling.py
python3 notify_slack.py

cd ..
python3 pipeline.py
python3 main.py \
  --kfp-api-url "http://kubeflow.ducdh.com/pipeline" \
  --kfp-dex-username "user@example.com" \
  --kfp-dex-password "12341234" \
  --kfp-dex-auth-type "local" \
  --kfp-namespace "kubeflow-user-example-com" \
  --cron-expr "0 * * * *" \
  --slack-channel "social" \
  --slack-bot-token "xoxb-9061740640727-9075485408565-1aTYM44VbkNYkUqnJCPrqYrS" \
  --pipeline-name "abc_xgb" \
  --experiment-name "abc_xgb" \
  --version-name "v2" \
  --job-name "demo-job-recurring-abc-xgb" \
  --minio-endpoint "minio.minio.svc.cluster.local:9000" \
  --minio-access-key "minio" \
  --minio-secret-key "minio123" \
  --bucket-name "sample-data" \
  --mlflow-endpoint "mlflow.mlflow.svc.cluster.local:5000" \
  --raw-train-object "data/application_train.csv" \
  --raw-test-object "data/application_test.csv" \
  --parent-run-name "xgb" \
  --n-features-to-select "5" \
  --iv-min "0.02" \
  --iv-max "0.4" \
  --missing-thres "0.1" \
  --model-type "xgb" \
  --suffix "test"
