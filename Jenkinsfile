pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'microwave1005/prediction-api'
        IMAGE_TAG    = 'latest'
    }

    stage('Test') {
        agent {
            docker {
                image 'microwave1005/test-runner:latest'
            }
        }
        steps {
            sh '''
                pip install pytest
                chmod +x src/client/test/test.sh
                src/client/test/test.sh
            '''
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
                    cd src/kfp

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

                // Promote to Staging
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

                input message: "Approve promotion of model ${modelName} to Production?"

                // Promote to Production
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
    }

        stage('Deploy') {
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

