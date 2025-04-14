# UnderWritingModel Project

## Overview
The UnderWritingModel project is designed to facilitate the development and evaluation of underwriting models using advanced machine learning techniques. The project leverages OptBinning for effective binning of categorical and numerical features, XGBoost as the primary training model, and SHAP for model interpretation. Additionally, MLflow is utilized for logging and tracking model performance.

## Project Structure
```
under_writing
├── src
│   ├── UnderWritingModel.py       # Contains the UnderWritingModel class
│   └── utils.py                    # Utility functions for data preprocessing
├── notebooks
│   └── under_writing_model_exploration.ipynb  # Jupyter notebook for model exploration
├── pyproject.toml                  # Project configuration and dependencies
├── README.md                       # Documentation for the project
└── .gitignore                      # Files and directories to ignore in Git
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can use the following commands:

```bash
git clone <repository-url>
cd under_writing
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Use the utility functions in `utils.py` for data preprocessing, including handling missing values and outlier detection.
2. **Model Training**: Instantiate the `UnderWritingModel` class from `UnderWritingModel.py`, fit the model on your training data, and make predictions.
3. **Model Evaluation**: Evaluate the model's performance using metrics logged with MLflow.
4. **Model Interpretation**: Use SHAP to explain the model's predictions and gain insights into feature importance.

## Example
Here is a brief example of how to use the `UnderWritingModel`:

```python
from src.UnderWritingModel import UnderWritingModel

# Initialize the model
model = UnderWritingModel()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Log metrics
model.log_metrics()

# Explain predictions
model.explain_predictions(X_test)
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## To-Do
- [ ] Add unit tests for `UnderWritingModel` and `utils.py`
- [ ] Refractor hardcode value in model_name in api.py 
- [ ] Integrate cross-validation into the training workflow
- [x] Add support for other models (e.g., LightGBM, CatBoost)
- [x] Create automated pipeline for model training and evaluation
- [ ] Enhance documentation with usage examples and API reference
- [ ] Add CI/CD pipeline using Jenkins
- [ ] Implement feature selection methods before model training
- [ ] Implement monitoring services with Grafana, Prometheus, Evidently, ELK 
- [ ] Insantiate Terraform for IaC to deploy in GCP k8s
- [ ] Move data in to GCP and using DVC for versioning

kubectl -n kubeflow port-forward pod/minio-6748f5ff9d-bx6v5 9000:9000
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
kubectl port-forward -n monitoring prometheus-kps-kube-prometheus-stack-prometheus-0 9090

# K8s - grafana - prometheus
kubectl --namespace monitoring get secrets kps-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo (admin - pwd: prom-operator)
kubectl --namespace monitoring port-forward $POD_NAME 3000

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

admin - KRvofPXWF7sooLZCK5gSdS

kubectl port-forward svc/cicd-jenkins 6060:6060 -n cicd

# mlflow
helm install mlflow community-charts/mlflow \
  --namespace mlflow \
  --create-namespace

helm uninstall mlflow -n mlflow
kubectl delete pods -n mlflow --selector=app.kubernetes.io/name=mlflow

kubectl port-forward svc/mlflow -n mlflow 5003:5000

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

kubectl delete pvc mlops-test-workspace -n kubeflow-user-example-com
