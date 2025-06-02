pipeline {
    agent any

    /* housekeeping */
    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    /* ========= PARAMETERS ========= */
    parameters {
        string (name: 'MODEL_NAME', defaultValue: 'xgb_underwriting')
        choice (name: 'MODEL_TYPE', choices: ['xgb','lgbm'])
        /* --- KFP recurring run --- */
        string (name: 'KFP_DEX_AUTH_TYPE', defaultValue: 'local')
        string (name: 'KFP_CRON_EXPR', defaultValue: '0 3 * * *')
        string (name: 'KUBEFLOW_NAMESPACE', defaultValue: 'kubeflow-user-example-com')
        /* --- MLflow run to deploy --- */
        string (name: 'MLFLOW_EXPERIMENT_NAME', defaultValue: 'Kubeflow Pipeline outside')
        string (name: 'MLFLOW_RUN_NAME'       , defaultValue: 'xgb_optuna_search')
    }

    /* ========= ENV ========= */
    environment {
        registry               = 'microwave1005/prediction-api'

        MLFLOW_TRACKING_URI    = 'http://mlflow.ducdh.com'
        MINIO_ENDPOINT         = 'minio.dhduc.com'
        MINIO_BUCKET_NAME      = 'sample-data'

        KFP_API_URL            = 'http://kubeflow.ducdh.com/pipeline'

        MINIO_CREDS            = credentials('minio-creds')
        AWS_ACCESS_KEY_ID      = "${MINIO_CREDS_USR}"
        AWS_SECRET_ACCESS_KEY  = "${MINIO_CREDS_PSW}"
        MLFLOW_S3_ENDPOINT_URL = "http://${MINIO_ENDPOINT}"

        TAG = "v.${env.BUILD_NUMBER}"

        CODE_CHANGED        = 'true'   
        NEED_PROMOTE        = 'true'
        IMAGE_EXISTS        = 'false'
        RUN_ID              = ''
    }

    stages {

        /* ---------------------------------------------------------- */
        stage('Detect changes & set flags') {
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }

            steps {
                script {
                    /* ---------- 1. code diff ---------- */
                    def changed = sh(returnStatus: true,
                                     script: 'git diff --quiet HEAD~1 HEAD') != 0
                    env.CODE_CHANGED = changed.toString()

                    /* ---------- 2. model promote? ---------- */
                    def needPromote = sh(
                        returnStatus: true,
                        script: """
                            python3 src/tools/is_new_model_needed.py \
                              --tracking-uri "${MLFLOW_TRACKING_URI}" \
                              --model-name   "${params.MODEL_NAME}" \
                              --stage        staging
                        """) == 0
                    env.NEED_PROMOTE = needPromote.toString()

                    /* ---------- 3. docker image exists? ---------- */
                    def imageExists = sh(
                        returnStatus: true,
                        script: """
                            docker manifest inspect ${registry}:${TAG} >/dev/null 2>&1
                        """) == 0
                    env.IMAGE_EXISTS = imageExists.toString()

                    /* ---------- 4. fetch run_id mlflow ---------- */
                    if (changed) {
                        env.RUN_ID = sh(
                            script: """
                                python3 src/tools/fetch_mlflow_run.py \
                                   --tracking-uri "${MLFLOW_TRACKING_URI}" \
                                   --experiment "${params.MLFLOW_EXPERIMENT_NAME}" \
                                   --run-name   "${params.MLFLOW_RUN_NAME}"
                            """, returnStdout: true).trim()
                    }

                }
            }
        }

        /* ---------------------------------------------------------- */
        stage('Unit tests + coverage') {
            when { expression { env.CODE_CHANGED == 'true' } }
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }
            steps {
                sh '''
                    PYTHONPATH=src pytest -m unittest tests/
                    echo "[INFO] Failing if coverage < 80%"
                    coverage report --fail-under=80
                '''
            }
        }

        /* ---------------------------------------------------------- */
        stage('Enable KFP recurring run') {
            steps {
                input message: "Approve KFP recurring run for ${params.PIPELINE_NAME}?"
            }
        }

        stage('Schedule KFP recurring run') {
            when { expression { env.CODE_CHANGED == 'true' } }
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'kubeflow-creds',
                    usernameVariable: 'KFP_DEX_USERNAME',
                    passwordVariable: 'KFP_DEX_PASSWORD')]) {
                    script {
                        def cronExpr = params.KFP_CRON_EXPR ?: '0 3 * * *'
                        dir('src') {
                            sh """
                                PYTHONPATH=. python3 tools/schedule_kfp_run.py \
                                    --kfp-api-url       "${KFP_API_URL}" \
                                    --kfp-dex-username  "${KFP_DEX_USERNAME}" \
                                    --kfp-dex-password  "${KFP_DEX_PASSWORD}" \
                                    --kfp-dex-auth-type "${params.KFP_DEX_AUTH_TYPE}" \
                                    --kfp-namespace     "${params.KUBEFLOW_NAMESPACE}" \
                                    --cron-expr         "${cronExpr}"
                            """
                        }
                    }
                }
            }
        }

        /* ---------------------------------------------------------- */
        stage('Promote to Staging') {
            when { expression { env.CODE_CHANGED == 'true' && env.NEED_PROMOTE == 'true' } }
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }
            steps {
                dir('src') {
                    sh """
                        python3 tools/promote_model.py \
                           --model       "${params.MODEL_NAME}" \
                           --from-stage  none \
                           --to-stage    staging \
                           --tracking-uri "${MLFLOW_TRACKING_URI}"
                    """
                }
            }
        }

        /* ---------------------------------------------------------- */
        stage('Build & Push Image') {
            when { expression { env.CODE_CHANGED == 'true' && env.IMAGE_EXISTS == 'false' } }
            steps {
                script {
                    echo "📦  Building image ${registry}:${TAG}"
                    def img = docker.build(
                        "${registry}:${TAG}",
                        "--build-arg MODEL_NAME=${params.MODEL_NAME} " +
                        "--build-arg MODEL_TYPE=${params.MODEL_TYPE} " +
                        "-f dockerfiles/Dockerfile.app ."
                    )
                    withCredentials([usernamePassword(
                        credentialsId: 'dockerhub-creds',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS')]) {
                        docker.withRegistry("https://${env.registry}", "${DOCKER_USER}:${DOCKER_PASS}") {
                            img.push(); img.push('latest')
                        }
                    }
                }
            }
        }

        /* ---------------------------------------------------------- */
        stage('Approve to Production') {
            when { expression { env.CODE_CHANGED == 'true' && env.NEED_PROMOTE == 'true' } }
            steps {
                input message: "Approve promotion of ${params.MODEL_NAME} to Production?"
            }
        }

        stage('Promote to Production') {
            when { expression { env.CODE_CHANGED == 'true' && env.NEED_PROMOTE == 'true' } }
            agent { docker { image 'microwave1005/kfp-jenkins-ci:latest' } }
            steps {
                dir('src') {
                    sh """
                        python3 tools/promote_model.py \
                           --model       "${params.MODEL_NAME}" \
                           --from-stage  staging \
                           --to-stage    production \
                           --tracking-uri "${MLFLOW_TRACKING_URI}"
                    """
                }
            }
        }

        /* ---------------------------------------------------------- */
        stage('Deploy to Google Kubernetes Engine') {
            agent { kubernetes { cloud 'prediction-api-gke' } }
            steps {
                script {
                    echo "[INFO] Deploying tag ${TAG} (run_id=${env.RUN_ID})"
                    sh """
                        helm upgrade --install api ./helm-charts/api \
                            --reuse-values \
                            --namespace api \
                            --set env.PARENT_RUN_ID=${env.RUN_ID} \
                            --set version=${TAG} \
                            --set monitoring.enabled=true \
                            --set image.tag=${TAG} \
                            --set replicaCount=1
                    """
                }
            }
        }
    }  /* end stages */

    /* ---------------------------------------------------------- */
    post {
        always   { echo '[INFO] Pipeline finished (success/abort/fail)' }
        cleanup  {
            sh 'docker image prune -f'
            echo '[INFO] Local Docker cache cleaned'
        }
    }
}
