pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    parameters {
        string(name: 'MODEL_NAME', defaultValue: 'v1_xgb_XGB', description: 'Model Name to Build & Promote')
        choice(name: 'MODEL_TYPE', choices: ['xgb','lgbm'], description: 'Model Type to use')
        string(name: 'MLFLOW_IP', defaultValue: '35.193.229.26', description: 'External IP of MLflow Ingress')
    }

    environment {
        registry               = 'microwave1005/prediction-api'
        registryCredential     = 'dockerhub-creds'
        MINIO_ENDPOINT         = 'minio.dhduc.com'
        MINIO_ACCESS_KEY       = 'minio'
        MINIO_SECRET_KEY       = 'minio123'
        MINIO_BUCKET_NAME      = 'sample-data'
        MLFLOW_TRACKING_URI    = 'http://mlflow.ducdh.com'

        CLUSTER_NAME           = 'prediction-platform'
        ZONE                   = 'us-central1-c'
        PROJECT_ID             = 'mlops-fsds'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            agent {
                docker {
                    image 'microwave1005/kfp-ci-jenkins'
                    reuseNode true
                }
            }
            steps {
                dir('testing') {
                    
                sh '''
                    cd testing
                    pytest -m unit

                    echo " Failing if coverage < 80%"
                    coverage report --fail-under=80
                '''
                }
            }
        }

        stage('Build') {
            steps {
                script {
                    echo "📦 Building image with MODEL_NAME=${params.MODEL_NAME}, MODEL_TYPE=${params.MODEL_TYPE}"
                    dockerImage = docker.build(
                        "${registry}:${BUILD_NUMBER}",
                        "--build-arg MODEL_NAME=${params.MODEL_NAME} --build-arg MODEL_TYPE=${params.MODEL_TYPE} -f dockerfiles/Dockerfile.app ."
                    )

                    echo '📤 Pushing image to Docker Hub...'
                    docker.withRegistry('', registryCredential) {
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Promote to Staging') {
            agent {
                docker {
                    image 'microwave1005/kfp-ci-jenkins'
                    args "--add-host mlflow.ducdh.com:${params.MLFLOW_IP}"
                    reuseNode true
                }
            }
            steps {
                sh """
                    echo "🌐 Verifying DNS & Host routing..."
                    curl -I http://mlflow.ducdh.com || echo "❌ DNS resolve failed"
                    curl -I -H 'Host: mlflow.ducdh.com' http://${params.MLFLOW_IP} || echo "❌ Host header routing failed"

                    echo "🚀 Promoting model to STAGING..."
                    python3 -c "import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri='http://mlflow.ducdh.com')
versions = client.get_latest_versions('${params.MODEL_NAME}', stages=['None'])
if versions:
    v = versions[0].version
    client.transition_model_version_stage('${params.MODEL_NAME}', v, 'Staging')
    print(f'✅ Promoted to Staging: ${params.MODEL_NAME} v{v}')
else:
    print('⚠️ No model version found.')"
                """
            }
        }

        stage('Approve to Production') {
            steps {
                input message: "Approve promotion of model ${params.MODEL_NAME} to Production?"
            }
        }

        stage('Promote to Production') {
            agent {
                docker {
                    image 'microwave1005/kfp-ci-jenkins'
                    args "--add-host mlflow.ducdh.com:${params.MLFLOW_IP}"
                    reuseNode true
                }
            }
            steps {
                sh """
                    echo "🌐 Verifying DNS & Host routing..."
                    curl -I http://mlflow.ducdh.com || echo "❌ DNS resolve failed"
                    curl -I -H 'Host: mlflow.ducdh.com' http://${params.MLFLOW_IP} || echo "❌ Host header routing failed"

                    echo "🚀 Promoting model to PRODUCTION..."
                    python3 -c "import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri='http://mlflow.ducdh.com')
versions = client.get_latest_versions('${params.MODEL_NAME}', stages=['Staging'])
if versions:
    v = versions[0].version
    client.transition_model_version_stage('${params.MODEL_NAME}', v, 'Production')
    print(f'✅ Promoted to Production: ${params.MODEL_NAME} v{v}')
else:
    print('⚠️ No Staging model found.')"
                """
            }
        }

        stage('Deploy to Google Kubernetes Engine') {
            steps {
                sh '''
                    set -e

                    echo "🔐 Authenticating to GCP..."
                    gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

                    echo "🔗 Fetching GKE credentials..."
                    gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID

                    echo "🚀 Upgrading API release with Helm..."
                    cd helm-charts/api

                    helm upgrade api . \
                      --namespace api \
                      --create-namespace \
                      --reuse-values \
                      --set monitoring.enabled=true \
                      --set image.tag=latest \
                      --set replicaCount=1 \
                      --set ingress.enabled=true \
                      --set ingress.rules[0].host=api.ducdh.com \
                      --set ingress.rules[0].paths[0].path="/" \
                      --set ingress.rules[0].paths[0].pathType=Prefix \
                      --set ingress.rules[0].paths[0].serviceName=prediction-api \
                      --set ingress.rules[0].paths[0].servicePort=8000
                '''
            }
        }
    }

    post {
        always {
            echo '✅ Pipeline execution complete.'
        }
        cleanup {
            echo '🧹 Cleaning up unused Docker images...'
            sh 'docker image prune -f'
        }
    }
}
