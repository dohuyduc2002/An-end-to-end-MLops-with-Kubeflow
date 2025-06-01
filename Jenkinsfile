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
        string(name: 'PIPELINE_NAME', defaultValue: 'kfp-outside-pipeline', description: 'Base run ID for KFP recurring run')
        string(name: 'PIPELINE_VERSION', defaultValue: 'v1', description: 'Base run ID for KFP recurring run')
        string(name: 'KFP_CRON_EXPR', defaultValue: '0 3 * * *', description: 'Cron expression for KFP recurring run')

        /* MLFlow config */
        string(name: 'MLFLOW_EXPERIMENT_NAME', defaultValue: 'Kubeflow Pipeline outside', description: 'MLFlow Experiment Name')
        string(name: 'MLFLOW_RUN_NAME', defaultValue: 'xgb_optuna_search', description: 'Model run name registered in MLFlow through Kubeflow Pipeline')
    }

    environment {
        /* Dockerhub config */
        registry           = 'microwave1005/prediction-api'

        /* MLflow config */
        MLFLOW_TRACKING_URI = 'http://mlflow.ducdh.com'

        /* MinIO config */
        MINIO_ENDPOINT      = 'minio.dhduc.com'
        MINIO_BUCKET_NAME   = 'sample-data'

        /*Kubeflow pipeline config */
        KFP_API_URL = 'http://kubeflow.ducdh.com/pipeline'

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
        // stage('Enable KFP recurring run') {
        //     agent {
        //         docker {
        //             image 'microwave1005/kfp-jenkins-ci:latest'
        //         }
        //     }
        //     steps {
        //         withCredentials([
        //             usernamePassword(credentialsId: 'kubeflow-creds', usernameVariable: 'KFP_DEX_USERNAME', passwordVariable: 'KFP_DEX_PASSWORD')
        //         ]) {
        //             script {
        //                 def cronExpr = params.KFP_CRON_EXPR ?: '0 3 * * *'
        //                 sh """
        //                     python3 src/schedule_kfp_run.py \
        //                         --kfp-api-url "${env.KFP_API_URL}" \
        //                         --kfp-dex-username "${KFP_DEX_USERNAME}" \
        //                         --kfp-dex-password "${KFP_DEX_PASSWORD}" \
        //                         --kfp-dex-auth-type "${params.KFP_DEX_AUTH_TYPE}" \
        //                         --pipeline-name "${params.PIPELINE_NAME}" \
        //                         --pipeline-version "${params.PIPELINE_VERSION}" \
        //                         --cron-expr "${cronExpr}"
        //                 """
        //             }
        //         }
        //     }
        // }

        stage('Build & Push Image') {
            steps {
                script {
                    echo "Building image MODEL_NAME=${params.MODEL_NAME}, MODEL_TYPE=${params.MODEL_TYPE}"
                    def tag = env.TAG
                    def imageName = "${env.registry}:${tag}"

                    def img = docker.build(
                        imageName,
                        "--build-arg MODEL_NAME=${params.MODEL_NAME} " +
                        "--build-arg MODEL_TYPE=${params.MODEL_TYPE} " +
                        "-f dockerfiles/Dockerfile.app ."
                    )

                    echo "Pushing image with tags: ${tag}, latest"

                    withCredentials([usernamePassword(credentialsId: 'dockerhub_creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                        docker.withRegistry("https://${env.registry}", "${DOCKER_USER}:${DOCKER_PASS}") {
                            img.push()
                        }

                        // Push "latest" tag separately
                        sh "docker tag ${imageName} ${env.registry}:latest"
                        sh "echo $DOCKER_PASS | docker login ${env.registry} -u $DOCKER_USER --password-stdin"
                        sh "docker push ${env.registry}:latest"
                    }
                }
            }
        }


        stage('Promote to Staging') {
            agent {
                docker {
                image 'microwave1005/kfp-jenkins-ci:latest'
                }
            }
            steps {
                script {
                sh """
                    python3 src/promote_model.py \
                    --model       "${params.MODEL_NAME}" \
                    --from-stage  none \
                    --to-stage    stagging \
                    --tracking-uri "${env.MLFLOW_TRACKING_URI}"
                """
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
                script {
                sh """
                    python3 src/promote_model.py \
                    --model       "${params.MODEL_NAME}" \
                    --from-stage  staging \
                    --to-stage    production \
                    --tracking-uri "${env.MLFLOW_TRACKING_URI}"
                """
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
                script {
                    // 1. Fetch the latest model version from MLFlow
                    def run_id = sh(
                        script: """
                            python3 src/fetch_mlflow_run.py \
                                --tracking-uri "${MLFLOW_TRACKING_URI}" \
                                --experiment-name "${params.MLFLOW_EXPERIMENT_NAME}" \
                                --run-name "${params.MLFLOW_RUN_NAME}" \
                        """,
                        returnStdout: true
                    ).trim()
                }
                echo "[INFO] Fetched run ID: ${run_id}"
                // 2. Set the PARENT_RUN_ID environment variable for Helm
                sh """
                    helm upgrade --install api ./helm-charts/api \
                        --reuse-values \
                        --namespace api \
                        --set env.PARENT_RUN_ID=${run_id} \
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
