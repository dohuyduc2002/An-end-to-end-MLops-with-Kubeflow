import argparse
from utils import KFPClientManager, create_recurring_run_with_params, get_runs_reponse

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


def schedule_recurring_run(kfp_client, cron_expr, namespace, params):
    run_info = get_runs_reponse(kfp_client, namespace=namespace)
    create_recurring_run_with_params(
        kfp_client=kfp_client,
        cron_expr=cron_expr,
        run_info=run_info,
        params=params,
    )


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
    parser.add_argument("--cron-expr", required=True, help="Cron expression")

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
        "--dest-train-object",
        default="preprocessed_train.csv",
        help="Destination train object",
    )
    parser.add_argument(
        "--dest-test-object",
        default="preprocessed_test.csv",
        help="Destination test object",
    )
    parser.add_argument(
        "--parent-run-name", default="xgb_optuna_search", help="Parent run name"
    )
    parser.add_argument(
        "--n-features-to-select", default="auto", help="Number of features to select"
    )
    parser.add_argument("--data-version", default="v1", help="Data version")
    parser.add_argument(
        "--model-name", default="xgb", choices=["xgb", "lgbm"], help="Model name"
    )
    parser.add_argument("--suffix", default="underwrite", help="Suffix")
    parser.add_argument(
        "--experiment-name", default="Underwriting_kfp", help="Experiment name"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    # Gom các params cho pipeline
    exclude = {
        "kfp_api_url",
        "kfp_dex_username",
        "kfp_dex_password",
        "kfp_dex_auth_type",
        "kfp_namespace",
        "cron_expr",
    }
    params = {k: v for k, v in vars(args).items() if k not in exclude}

    kfp_client = authenticate_kfp_client(args)
    print("✅ Authenticated KFP client created.")
    schedule_recurring_run(
        kfp_client=kfp_client,
        cron_expr=args.cron_expr,
        namespace=args.kfp_namespace,
        params=params,
    )


if __name__ == "__main__":
    main()
