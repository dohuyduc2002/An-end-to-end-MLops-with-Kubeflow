pipeline {
    agent any
    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    parameters {
        /* The model name will be model_name + suffix in `src/kfp_outside/main.py` */
        string(name: 'MODEL_NAME', defaultValue: 'xgb_underwriting', description: 'Model Name to Build & Promote')
        choice(name: 'MODEL_TYPE', choices: ['xgb','lgbm'], description: 'Model implementation')

        /* KFP config */
        string(name: 'KFP_DEX_AUTH_TYPE', defaultValue: 'local', description: 'Kubeflow Dex Auth Type')

        /* Recurring job config */
        string(name: 'BASE_RUN_ID', defaultValue: 'b4a73df0-cac0-4bbb-8d57-55612c32ae43', description: 'Run ID of KFP pipeline to convert to recurring run')
        string(name: 'KFP_CRON_EXPR', defaultValue: '0 3 * * *', description: 'Cron expression for KFP recurring run')
    }

    environment {
        /* Dockerhub config */
        registry           = 'microwave1005/prediction-api'
        dockerhub_credential = 'dockerhub-creds'

        /* MLflow config */
        MLFLOW_TRACKING_URI = 'http://mlflow.ducdh.com'

        /* MinIO config */
        MINIO_ENDPOINT      = 'minio.dhduc.com'
        MINIO_BUCKET_NAME   = 'sample-data'

        /*Kubeflow pipeline config */
        KFP_API_URL = 'http://kubeflow.ducdh.com/pipeline'
        kubeflow_credential = 'kubeflow-creds'

        /*Minio config for mlflow artifact store */ 
        MINIO_CREDS = credentials('minio-creds')
        AWS_ACCESS_KEY_ID      = "${MINIO_CREDS_USR}"
        AWS_SECRET_ACCESS_KEY  = "${MINIO_CREDS_PSW}"
        MLFLOW_S3_ENDPOINT_URL = "http://${MINIO_ENDPOINT}"

        TAG = "v.${env.BUILD_NUMBER}"
    }

    stages {

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
        stage('Enable KFP recurring run') {
            agent {
                docker {
                    image 'microwave1005/kfp-jenkins-ci:latest'
                }
            }
            steps {
                withCredentials([
                    usernamePassword(credentialsId: 'kubeflow-creds', usernameVariable: 'KFP_DEX_USERNAME', passwordVariable: 'KFP_DEX_PASSWORD')
                ]) {
                    script {
                        def cronExpr = params.KFP_CRON_EXPR ?: '0 3 * * *'
                        sh """
                            python3 src/schedule_kfp_run.py \
                                --kfp-api-url "${env.KFP_API_URL}" \
                                --kfp-dex-username "${KFP_DEX_USERNAME}" \
                                --kfp-dex-password "${KFP_DEX_PASSWORD}" \
                                --kfp-dex-auth-type "${params.KFP_DEX_AUTH_TYPE}" \
                                --run-id "${params.BASE_RUN_ID}" \
                                --cron-expr "${cronExpr}"
                        """
                    }
                }
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
                    docker.withRegistry('', env.dockerhub_credential) {
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
                withEnv([
                    "AWS_ACCESS_KEY_ID=${env.AWS_ACCESS_KEY_ID}",
                    "AWS_SECRET_ACCESS_KEY=${env.AWS_SECRET_ACCESS_KEY}",
                    "MLFLOW_S3_ENDPOINT_URL=${env.MLFLOW_S3_ENDPOINT_URL}",
                    "MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI}",
                    "MODEL_NAME=${params.MODEL_NAME}"
                ]) {
                    sh '''
                        python3 src/promote_model.py \
                            --model       "${MODEL_NAME}" \
                            --from-stage  none \
                            --to-stage    staging \
                            --tracking-uri "${MLFLOW_TRACKING_URI}"
                    '''
                }
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
                withEnv([
                    "AWS_ACCESS_KEY_ID=${env.AWS_ACCESS_KEY_ID}",
                    "AWS_SECRET_ACCESS_KEY=${env.AWS_SECRET_ACCESS_KEY}",
                    "MLFLOW_S3_ENDPOINT_URL=${env.MLFLOW_S3_ENDPOINT_URL}",
                    "MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI}",
                    "MODEL_NAME=${params.MODEL_NAME}"
                ]) {
                    sh '''
                        python3 src/promote_model.py \
                            --model       "${MODEL_NAME}" \
                            --from-stage  staging \
                            --to-stage    production \
                            --tracking-uri "${MLFLOW_TRACKING_URI}"
                    '''
                }
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
        always {
            echo '[INFO] Pipeline execution complete.'
        }
        cleanup {
            sh 'docker image prune -f'
            echo '[INFO] Docker images cleaned.'
        }
    }
}
