import argparse
from utils import upload_pipeline_and_version, KFPClientManager
import logging
from datetime import datetime

# Due to running Kubeflow Pipeline outside cluster, we need to insantiate client through ClientManager class
# This class will handle authentication and client creation by disable TLS, and get session cookies
def authenticate_kfp_client(args):
    client_auth_manager = KFPClientManager(
        api_url=args.kfp_api_url,
        dex_username=args.kfp_dex_username,
        dex_password=args.kfp_dex_password,
        dex_auth_type=args.kfp_dex_auth_type,
        skip_tls_verify=True,
    )
    return client_auth_manager.create_kfp_client()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Authenticate KFP and schedule recurring pipeline run."
    )    
    parser.add_argument(
        "--kfp-api-url", required=True, help="Kubeflow Pipelines API URL"
    )
    parser.add_argument("--kfp-dex-username", required=True, help="Dex username")
    parser.add_argument("--kfp-dex-password", required=True, help="Dex password")
    parser.add_argument("--kfp-dex-auth-type", required=True, help="Dex auth type")
    parser.add_argument("--kfp-namespace", required=True, help="KFP namespace")
    parser.add_argument("--cron-expr",required=True, help="Go Cron expression")
    
    parser.add_argument("--slack-channel", required=True, help="Slack channel")
    parser.add_argument("--slack-bot-token", required=True, help="Slack bot token")
    
    parser.add_argument("--pipeline-name", required=True, help="Pipeline name")
    parser.add_argument("--experiment-name", required=True, help="Experiment name")
    parser.add_argument("--version-name", required=True, help="Pipeline version name")
    parser.add_argument("--job-name",required=True, help="Job name")

    parser.add_argument("--minio-endpoint", required=True, help="MinIO endpoint")
    parser.add_argument("--minio-access-key", required=True, help="MinIO access key")
    parser.add_argument("--minio-secret-key", required=True, help="MinIO secret key")
    parser.add_argument("--bucket-name", required=True, help="Bucket name")
    parser.add_argument("--mlflow-endpoint", required=True, help="MLflow endpoint")
    
    parser.add_argument(
        "--raw-train-object",
        default="data/application_train.csv",
        help="Raw train object path",
    )
    parser.add_argument(
        "--raw-test-object",
        default="data/application_test.csv",
        help="Raw test object path",
    )
    parser.add_argument(
        "--parent-run-name", required=True, help="Parent run name"
    )
    parser.add_argument(
        "--n-features-to-select", required=True, help="Number of features to select"
    )
    parser.add_argument(
        "--iv-min", type=float, required=True, help="Minimum IV value"
    )
    parser.add_argument(
        "--iv-max", type=float, required=True, help="Maximum IV value"
    )
    parser.add_argument(
        "--missing-thres", type=float, required=True, help="Missing threshold"
    )
    parser.add_argument(
        "--model-type", required=True, choices=["xgb", "lgbm"], help="Model type"
    )
    parser.add_argument("--suffix", default="underwrite", help="Suffix")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    exclude = {
        "kfp_api_url",
        "kfp_dex_username",
        "kfp_dex_password",
        "kfp_dex_auth_type",
        "kfp_namespace",
        "cron_expr",
        "pipeline_name",
        "version_name",
    }
    params = {k: v for k, v in vars(args).items() if k not in exclude}

    kfp_client = authenticate_kfp_client(args)

    pipeline_yaml = "pipeline.yaml"

    # Upload pipeline/version and get IDs
    pipeline_id, version_id, version_name = upload_pipeline_and_version(
        kfp_client, pipeline_yaml, args.pipeline_name, args.version_name, args.kfp_namespace
    )
    experiment = kfp_client.create_experiment(name=args.experiment_name, namespace=args.kfp_namespace)
    experiment_id = experiment.experiment_id

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{args.job_name} at {now_str}"

    # if args.set_run_recurring.lower() == "true":
    run = kfp_client.create_recurring_run(
        experiment_id=experiment_id,
        job_name=job_name,
        cron_expression=args.cron_expr,
        pipeline_id=pipeline_id,
        version_id=version_id,
        params=params,
        enabled=True,
        description=args.job_name,
        enable_caching=None
    )
    logging.info(run)
    logging.info("Recurring run scheduled successfully.")

    # else: 
    #     run = kfp_client.run_pipeline(
    #         experiment_id=experiment_id,
    #         job_name=job_name,
    #         pipeline_id=pipeline_id,
    #         version_id=version_id,
    #         params=params,
    #     )
    #     logging.info(run)
    #     logging.info("Pipeline run one-off started successfully.")


if __name__ == "__main__":
    main()
