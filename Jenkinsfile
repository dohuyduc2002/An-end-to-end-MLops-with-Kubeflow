pipeline {
    agent any
    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    parameters {
        string (name: 'MODEL_NAME', defaultValue: 'v1_xgb_XGB', description: 'Model Name to Build & Promote')
        choice (name: 'MODEL_TYPE', choices: ['xgb','lgbm'],        description: 'Model implementation')
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
    }

    stages {

        stage('Test') {
            agent { 
                docker { 
                    image 'microwave1005/kfp-ci-jenkins' 
                        } 
                    }
            steps {
                dir('tests') {
                    sh '''
                        pytest -m unittest
                        echo "[INFO] Failing if coverage < 80%"
                        coverage report --fail-under=80
                    '''
                }
            }
        }

        stage('Build') {
            steps {
                script {
                    echo "Building image MODEL_NAME=${params.MODEL_NAME}, MODEL_TYPE=${params.MODEL_TYPE}"
                    def img = docker.build(
                        "${registry}:${BUILD_NUMBER}",
                        "--build-arg MODEL_NAME=${params.MODEL_NAME} " +
                        "--build-arg MODEL_TYPE=${params.MODEL_TYPE} " +
                        "-f dockerfiles/Dockerfile.app ."
                    )
                    echo '[INFO] Pushing image to Docker Hub…'
                    docker.withRegistry('', registryCredential) {
                        img.push()
                        img.push('latest')
                    }
                }
            }
        }

        stage('Promote to Staging') {
            agent { 
                docker { 
                    image 'microwave1005/kfp-ci-jenkins'
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
                    image 'microwave1005/kfp-ci-jenkins'
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
            steps {
                sh '''
                    set -e
                    echo "[INFO] Authenticating to GCP…"
                    gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

                    echo "[INFO] Fetching cluster creds…"
                    gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID

                    echo "[INFO] Rolling upgrade with Helm…"
                    cd helm-charts/api
                    helm upgrade api . \
                      --namespace api --create-namespace --reuse-values \
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
        always  { echo '[INFO] Pipeline execution complete.' }
        cleanup { sh 'docker image prune -f'; echo '[INFO] Docker images cleaned.' }
    }
}
