pipeline {
    agent any
    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    parameters {
        string(name: 'MODEL_NAME', defaultValue: 'v1_xgb_XGB', description: 'Model Name to Build & Promote')
        choice(name: 'MODEL_TYPE', choices: ['xgb','lgbm'], description: 'Model implementation')
    }

    environment {
        /* Dockerhub config */
        registry           = 'microwave1005/prediction-api'
        registryCredential = 'dockerhub-creds'

        /* GKE config */
        CLUSTER_NAME       = 'prediction-platform'
        ZONE               = 'us-central1-c'
        PROJECT_ID         = 'mlops-fsds'

        /* MLflow config */
        MLFLOW_TRACKING_URI = 'http://mlflow.ducdh.com'

        /* MinIO config */
        MINIO_ENDPOINT      = 'minio.dhduc.com'
        MINIO_BUCKET_NAME   = 'sample-data'
         
        MINIO_CREDS = credentials('minio-creds')
        AWS_ACCESS_KEY_ID      = "${MINIO_CREDS_USR}"
        AWS_SECRET_ACCESS_KEY  = "${MINIO_CREDS_PSW}"
        MLFLOW_S3_ENDPOINT_URL = "http://${MINIO_ENDPOINT}"

        TAG = "v.${env.BUILD_NUMBER}"
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
                    image 'microwave1005/kfp-jenkins-ci:latest'
                } 
            }
            steps {
                sh '''
                    PYTHONPATH=src pytest -m unittest tests/
                    echo "[INFO] Failing if coverage < 80%"
                    coverage report --fail-under=80
                '''
            }
        }

        stage('Build & Push Image') {
            steps {
                script {
                    echo "Building image MODEL_NAME=${params.MODEL_NAME}, MODEL_TYPE=${params.MODEL_TYPE}"
                    def tag = env.TAG
                    def img = docker.build(
                        "${env.registry}:${tag}",
                        "--build-arg MODEL_NAME=${params.MODEL_NAME} " +
                        "--build-arg MODEL_TYPE=${params.MODEL_TYPE} " +
                        "-f dockerfiles/Dockerfile.app ."
                    )
                    echo "Pushing image with tags: ${tag}, latest"
                    docker.withRegistry('', env.registryCredential) {
                        img.push()
                        sh "docker tag ${env.registry}:${tag} ${env.registry}:latest"
                        sh "docker push ${env.registry}:latest"
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
                sh '''
                    python3 src/promote_model.py \
                        --model       "${MODEL_NAME}" \
                        --from-stage  none \
                        --to-stage    staging \
                        --tracking-uri "${MLFLOW_TRACKING_URI}"
                '''
            }
        }

        stage('Approve to Production') {
            steps {
                input message: "Approve promotion of ${params.MODEL_NAME} to Production?"
            }
        }

        stage('Promote to Production') {
            agent { 
                docker { 
                    image 'microwave1005/kfp-jenkins-ci:latest'
                }
            }
            steps {
                sh '''
                    python3 src/promote_model.py \
                        --model       "${MODEL_NAME}" \
                        --from-stage  staging \
                        --to-stage    production \
                        --tracking-uri "${MLFLOW_TRACKING_URI}"
                '''
            }
        }

        stage('Deploy to Google Kubernetes Engine') {
            agent {
                kubernetes {
                    cloud 'prediction-api-gke'
                }
            }
            steps {
                sh """
                    helm upgrade --install api ./helm-charts/api \
                        --reuse-values \
                        --namespace api \
                        --set version=${TAG} \
                        --set monitoring.enabled=true \
                        --set image.tag=${TAG} \
                        --set replicaCount=1
                """
            }
        }
    }

    post {
        always  { echo '[INFO] Pipeline execution complete.' }
        cleanup { sh 'docker image prune -f'; echo '[INFO] Docker images cleaned.' }
    }
}
