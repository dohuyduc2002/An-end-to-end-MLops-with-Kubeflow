import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kfp_outside.utils import (
    KFPClientManager,
    create_recurring_run_with_params,
    get_runs_reponse,
)


def authenticate_kfp_client(args):
    client_auth_manager = KFPClientManager(
        api_url=args.kfp_api_url,
        dex_username=args.kfp_dex_username,
        dex_password=args.kfp_dex_password,
        dex_auth_type=args.kfp_dex_auth_type,
        skip_tls_verify=True,
    )
    return client_auth_manager.create_kfp_client()


def schedule_recurring_run(kfp_client, cron_expr, namespace):
    run_info = get_runs_reponse(kfp_client, namespace=namespace)
    create_recurring_run_with_params(
        kfp_client=kfp_client,
        cron_expr=cron_expr,
        run_info=run_info,
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

    return parser.parse_args()


def main():
    args = parse_arguments()
    kfp_client = authenticate_kfp_client(args)
    print("✅ Authenticated KFP client created.")
    schedule_recurring_run(
        kfp_client=kfp_client,
        cron_expr=args.cron_expr,
        namespace=args.kfp_namespace,
    )


if __name__ == "__main__":
    main()
