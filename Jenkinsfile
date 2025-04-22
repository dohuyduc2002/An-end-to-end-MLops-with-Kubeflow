pipeline {
    agent any
    environment {
        // Name of the Docker image and tag
        DOCKER_IMAGE = 'microwave1005/prediction-api'
        IMAGE_TAG   = 'latest'
    }
    stages {
        stage('Test') {
            steps {
                sh './test/test.sh'
            }
        }
        stage('Build & Push') {
            agent {
                docker {
                    image 'docker:latest'
                    args '-v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            steps {
                sh '''
                    set -o allexport
                    source ../.env
                    set +o allexport

                    # Login to Docker registry using credentials from .env
                    docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD

                    # Build Docker image and push to registry
                    docker build -t ${DOCKER_IMAGE}:${IMAGE_TAG} .
                    docker push ${DOCKER_IMAGE}:${IMAGE_TAG}
                '''
            }
        }
        stage('Approve') {
            steps {
                script {
                    def modelName = input(
                        id: 'modelApproval', message: 'Model Promotion Approval',
                        parameters: [string(defaultValue: 'v2_XGB', description: 'Model Name to Promote', name: 'modelName')]
                    )

                    // Promote model to Staging
                    sh """
                        python3 -c '
import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri="http://localhost:5003")
versions = client.get_latest_versions("${modelName}", stages=["None"])
if versions:
    v = versions[0].version
    client.transition_model_version_stage("${modelName}", v, "Staging")
    print(f"Model {modelName} version {v} promoted to Staging")
else:
    print("No model version found in None stage.")
'
                    """

                    // Wait for manual approval before promoting to Production
                    input message: "Approve promotion of model ${modelName} to Production?"

                    // Promote model to Production
                    sh """
                        python3 -c '
import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri="http://localhost:5003")
versions = client.get_latest_versions("${modelName}", stages=["Staging"])
if versions:
    v = versions[0].version
    client.transition_model_version_stage("${modelName}", v, "Production")
    print(f"Model {modelName} version {v} promoted to Production")
else:
    print("No model version found in Staging stage.")
'
                    """
                }
            }
        }
        stage('Deploy') {
            steps {
                sh '''
                    # Deploy API by applying Kubernetes manifests from k8s directory
                    cd ..
                    cd k8s
                    kubectl apply -f .

                    # Wait for deployment rollout to complete
                    echo "Waiting for deployment to complete..."
                    kubectl rollout status deployment/prediction-api -n monitoring

                    # Check service and endpoint
                    kubectl get svc -n monitoring

                    # Optional: set up ingress or test endpoint with curl
                    echo "Deployment done."
                '''
            }
        }
    }
    post {
        always {
            echo 'Pipeline execution complete.'
        }
    }
}
