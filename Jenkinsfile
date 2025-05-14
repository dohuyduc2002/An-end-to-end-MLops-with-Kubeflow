pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    parameters {
        string(name: 'MODEL_NAME', defaultValue: 'v1_xgb_XGB', description: 'Model Name to Build & Promote')
        string(name: 'MODEL_TYPE', defaultValue: 'xgb', description: 'Model Type (e.g. xgboost, rf)')
    }

    environment {
        registry               = 'microwave1005/prediction-api'
        registryCredential     = 'dockerhub-creds'
        MINIO_ENDPOINT         = 'minio.dhduc.com'
        MINIO_ACCESS_KEY       = 'minio'
        MINIO_SECRET_KEY       = 'minio123'
        MINIO_BUCKET_NAME      = 'sample-data'
        MLFLOW_TRACKING_URI    = 'http://mlflow.ducdh.com'
        KFP_API_URL            = 'http://kubeflow.ducdh.com/pipeline'
        KFP_DEX_USERNAME       = 'user@example.com'
        KFP_DEX_PASSWORD       = '12341234'
        KFP_SKIP_TLS_VERIFY    = 'False'
        KFP_DEX_AUTH_TYPE      = 'local'
    }

    stages {

        stage('Build') {
            steps {
                script {
                    echo "Building image with MODEL_NAME=${params.MODEL_NAME}, MODEL_TYPE=${params.MODEL_TYPE}"
                    dockerImage = docker.build(
                        "${registry}:${BUILD_NUMBER}",
                        "--build-arg MODEL_NAME=${params.MODEL_NAME} --build-arg MODEL_TYPE=${params.MODEL_TYPE} -f dockerfiles/Dockerfile.app ."
                    )

                    echo 'Pushing image to Docker Hub...'
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
                    image 'microwave1005/kfp-ci-jenkins:latest'
                }
            }
            steps {
                script {
                    sh """
                        python3 -c "import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri='${MLFLOW_TRACKING_URI}')
versions = client.get_latest_versions('${params.MODEL_NAME}', stages=['None'])
if versions:
    v = versions[0].version
    client.transition_model_version_stage('${params.MODEL_NAME}', v, 'Staging')
    print(f'Promoted to Staging: {params.MODEL_NAME} v{v}')
else:
    print('No model version found.')"
                    """
                }
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
                    image 'microwave1005/kfp-ci-jenkins:latest'
                }
            }
            steps {
                sh """
                    python3 -c "import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri='${MLFLOW_TRACKING_URI}')
versions = client.get_latest_versions('${params.MODEL_NAME}', stages=['Staging'])
if versions:
    v = versions[0].version
    client.transition_model_version_stage('${params.MODEL_NAME}', v, 'Production')
    print(f'Promoted to Production: ${params.MODEL_NAME} v{v}')
else:
    print('No Staging model found.')"
                """
            }
        }

        stage('Deploy') {
            agent {
                docker {
                    image 'microwave1005/kfp-ci-jenkins:latest'
                    args '-u root'
                }
            }
            steps {
                sh '''
                    cd k8s
                    kubectl apply -f .

                    echo "Waiting for deployment..."
                    kubectl rollout status deployment/prediction-api -n monitoring

                    kubectl get svc -n monitoring
                    echo "Deployment done."
                '''
            }
        }
    }

    post {
        always {
            echo '✅ Pipeline execution complete.'
        }
    }
}
