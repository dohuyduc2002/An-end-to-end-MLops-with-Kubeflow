pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    parameters {
        string(name: 'KFP-DEX-AUTH-TYPE', defaultValue: 'local')
        string(name: 'KUBEFLOW-NAMESPACE', defaultValue: 'kubeflow-user-example-com')
        string(name: 'cron-expr', defaultValue: '0 * * * *') /* robfig cron expression */
        string(name: 'pipeline-name', defaultValue: 'underwrite-pipeline')
        string(name: 'experiment-name', defaultValue: 'underwrite-experiment')  /* This also being used in fetch mlflow run id */ 
        string(name: 'version-name', defaultValue: 'v1')
        string(name: 'job-name', defaultValue: 'underwrite-job')
        string(name: 'raw-train-object', defaultValue: 'data/application_train.csv')
        string(name: 'raw-test-object', defaultValue: 'data/application_test.csv')
        string(name: 'parent-run-name', defaultValue: 'xgb_optuna_search') /* This also being used in fetch mlflow run id */ 
        string(name: 'n-features-to-select', defaultValue: 'auto')
        string(name: 'iv-min', defaultValue: '0.02')
        string(name: 'iv-max', defaultValue: '0.5')
        string(name: 'missing-thres', defaultValue: '0.5')
        choice(name: 'model-type', choices: ['xgb', 'lgbm']) 
        string(name: 'suffix', defaultValue: 'underwrite')
        string(name: 'MLFLOW_REGISTERED_MODEL_NAME', defaultValue: 'xgb_underwrite')
    }

    environment {
        registry              = 'microwave1005/prediction-api'
        registryCredential    = 'dockerhub-creds'
        TAG                   = "${BUILD_NUMBER}"

        MLFLOW_TRACKING_URI   = 'http://mlflow.ducdh.com'
        MINIO_ENDPOINT        = 'minio.dhduc.com'
        MINIO_BUCKET_NAME     = 'sample-data'
        KFP_API_URL           = 'http://kubeflow.ducdh.com/pipeline'
        EVIDENTLY_WORKSPACE   = 'http://evidently.dhduc.com:8000'

        MINIO_CREDS           = credentials('minio-creds')
        AWS_ACCESS_KEY_ID     = "${MINIO_CREDS_USR}"
        AWS_SECRET_ACCESS_KEY = "${MINIO_CREDS_PSW}"
        MLFLOW_S3_ENDPOINT_URL = "http://${MINIO_ENDPOINT}"

        RUN_ID = ''
    }

    stages {

        stage('Unit tests + coverage') {
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }
            steps {
                dir('tests') {
                    sh '''
                        pytest
                        echo "[INFO] Failing if coverage < 80%"
                        coverage report --fail-under=80
                    '''
                }
            }
        }

        stage('Schedule KFP recurring run') {
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'kubeflow-creds',
                        usernameVariable: 'KFP_DEX_USERNAME',
                        passwordVariable: 'KFP_DEX_PASSWORD'
                    ),
                    string(
                        credentialsId: 'slackbot',
                        variable: 'SLACK_BOT_TOKEN'
                    )
                ]) {
                    script {
                        def cronExpr = params['cron-expr']
                        dir('src/pipeline') {
                            sh """
                                PYTHONPATH=. python3 main.py \
                                    --kfp-api-url       "${KFP_API_URL}" \
                                    --kfp-dex-username  "${KFP_DEX_USERNAME}" \
                                    --kfp-dex-password  "${KFP_DEX_PASSWORD}" \
                                    --kfp-dex-auth-type "${params['KFP-DEX-AUTH-TYPE']}" \
                                    --kfp-namespace     "${params['KUBEFLOW-NAMESPACE']}" \
                                    --cron-expr         "${cronExpr}" \
                                    --slack-channel     "${params['slack-channel']}" \
                                    --slack-bot-token   "${SLACK_BOT_TOKEN}" \
                                    --pipeline-name     "${params['pipeline-name']}" \
                                    --experiment-name   "${params['experiment-name']}" \
                                    --version-name      "${params['version-name']}" \
                                    --job-name          "${params['job-name']}" \
                                    --minio-endpoint    "${MINIO_ENDPOINT}" \
                                    --minio-access-key  "${AWS_ACCESS_KEY_ID}" \
                                    --minio-secret-key  "${AWS_SECRET_ACCESS_KEY}" \
                                    --bucket-name       "${MINIO_BUCKET_NAME}" \
                                    --mlflow-endpoint   "${MLFLOW_TRACKING_URI}" \
                                    --raw-train-object  "${params['raw-train-object']}" \
                                    --raw-test-object   "${params['raw-test-object']}" \
                                    --parent-run-name   "${params['parent-run-name']}" \
                                    --n-features-to-select "${params['n-features-to-select']}" \
                                    --iv-min            "${params['iv-min']}" \
                                    --iv-max            "${params['iv-max']}" \
                                    --missing-thres     "${params['missing-thres']}" \
                                    --model-type        "${params['model-type']}" \
                                    --suffix            "${params['suffix']}"
                            """
                        }
                    }
                }
            }
        }

        stage('Build & Push Image') {
            steps {
                script {
                    echo "[INFO] Building image for deployment..."
                    def dockerImage = docker.build(
                        "${registry}:${BUILD_NUMBER}",
                        "--build-arg MODEL_NAME=${params.MLFLOW_REGISTERED_MODEL_NAME} " +
                        "--build-arg MODEL_TYPE=${params['model-type']} " +
                        "-f dockerfiles/Dockerfile.app ."
                    )

                    echo "[INFO] Pushing Docker image to Docker Hub..."
                    docker.withRegistry('', registryCredential) {
                        dockerImage.push("${BUILD_NUMBER}")
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Approve to Production') {
            steps {
                input message: "Approve promotion of ${params.MLFLOW_REGISTERED_MODEL_NAME} to Production?"
            }
        }

        stage('Promote to Production') {
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }
            steps {
                script {
                    dir('src') {
                        sh """
                            python3 tools/promote_model.py \
                                --model        "${params.MLFLOW_REGISTERED_MODEL_NAME}" \
                                --from-stage   staging \
                                --to-stage     production \
                                --tracking-uri "${MLFLOW_TRACKING_URI}"
                        """
                    }
                }
            }
        }

        stage('Fetch Mlflow run_id') {
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }
            steps {
                script {
                    env.RUN_ID = sh(
                        script: """
                            python3 src/tools/fetch_mlflow_run.py \
                                --tracking-uri "${MLFLOW_TRACKING_URI}" \
                                --experiment   "${params['experiment-name']}" \
                                --run-name     "${params['parent-run-name']}"
                        """,
                        returnStdout: true
                    ).trim()
                }
            }
        }

        stage('Deploy to Google Kubernetes Engine') {
            agent {
                kubernetes {
                    cloud 'prediction-api-gke'
                    yaml """
                    apiVersion: v1
                    kind: Pod
                    metadata:
                      labels:
                        jenkins-agent: gke-deploy
                    spec:
                      containers:
                      - name: helm
                        image: microwave1005/gke-helm-agent:latest
                        imagePullPolicy: Always
                        tty: true
                        volumeMounts:
                        - name: gcp-key
                          mountPath: /secrets
                          readOnly: true
                      volumes:
                      - name: gcp-key
                        secret:
                          secretName: gcp-key
                    """
                }
            }
            steps {
                script {
                    container('helm') {
                        sh '''
                            echo "[INFO] Deploying tag ${TAG} (run_id=${RUN_ID})"

                            gcloud auth activate-service-account --key-file=/secrets/gcp-key.json
                            gcloud config set project mlops-fsds
                            gcloud container clusters get-credentials prediction-platform --zone us-central1-c

                            helm upgrade --install api ./helm-charts/api \
                                --namespace api \
                                --set monitoring.enabled=true \
                                --set replicaCount=1 \
                                --set env.EVIDENTLY_WORKSPACE=${EVIDENTLY_WORKSPACE} \
                                --set env.PARENT_RUN_ID=${RUN_ID} \
                                --set version=${TAG} \
                                --set image.tag=${TAG} \
                                --set ingress.enabled=true \
                                --set ingress.enabled=true \
                                --set ingress.rules[0].host=api.ducdh.com \
                                --set ingress.rules[0].paths[0].path="/" \
                                --set ingress.rules[0].paths[0].pathType=Prefix \
                                --set ingress.rules[0].paths[0].serviceName=prediction-api \
                                --set ingress.rules[0].paths[0].servicePort=8000 \
                        '''
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                echo '[INFO] Pipeline finished (success/abort/fail)'
            }
        }
        cleanup {
            script {
                sh 'docker image prune -f'
                echo '[INFO] Local Docker cache cleaned'
            }
        }
    }
}
