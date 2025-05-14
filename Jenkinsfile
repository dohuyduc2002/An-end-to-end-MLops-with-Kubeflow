pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    environment {
        registry = 'microwave1005/prediction-api'
        registryCredential = 'dockerhub-creds'
        MINIO_ENDPOINT        = 'minio.dhduc.com'
        MINIO_ACCESS_KEY      = 'minio'
        MINIO_SECRET_KEY      = 'minio123'
        MINIO_BUCKET_NAME     = 'sample-data'
        MLFLOW_TRACKING_URI   = 'http://mlflow.ducdh.com'
        KFP_API_URL           = 'http://kubeflow.ducdh.com/pipeline'
        KFP_DEX_USERNAME      = 'user@example.com'
        KFP_DEX_PASSWORD      = '12341234'
        KFP_SKIP_TLS_VERIFY   = 'False'
        KFP_DEX_AUTH_TYPE     = 'local'
    }
    /*
    stages {
        stage('Test') {
            agent {
                docker {
                    image 'microwave1005/kfp-ci-jenkins:latest'
                }
            }
            steps {
                checkout scm
                sh '''
                    echo "Checking workspace"
                    ls -R
                    echo "Running tests"
                    cd testing
                    chmod +x test.sh
                    ./test.sh
                '''
            }
        }
    */
        stage('Build') {
            steps {
                script {
                    echo 'Building image for deployment...'
                    dockerImage = docker.build("${registry}:${BUILD_NUMBER}", "dockerfiles --file Dockerfile.app")
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
                    env.modelName = input(
                        id: 'modelApproval', message: 'Model Promotion Approval',
                        parameters: [string(defaultValue: 'v1_xgb_XGB', description: 'Model Name to Promote', name: 'modelName')]
                    )

                    sh """
                        python3 -c "import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri='http://mlflow.mlflow.svc:5000')
versions = client.get_latest_versions('${modelName}', stages=['None'])
if versions:
    v = versions[0].version
    client.transition_model_version_stage('${modelName}', v, 'Staging')
    print(f'Promoted to Staging: {modelName} v{v}')
else:
    print('No model version found.')"
                    """
                }
            }
        }

        stage('Approve to Production') {
            steps {
                input message: "Approve promotion of model ${modelName} to Production?"
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
client = mlflow.tracking.MlflowClient(tracking_uri='http://mlflow.mlflow.svc:5000')
versions = client.get_latest_versions('${modelName}', stages=['Staging'])
if versions:
    v = versions[0].version
    client.transition_model_version_stage('${modelName}', v, 'Production')
    print(f'Promoted to Production: {modelName} v{v}')
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
            echo 'Pipeline execution complete.'
        }
    }

