import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.kfp_outside.utils import (
    KFPClientManager,
    get_or_upload_pipeline,
    create_recurring_run,
    get_latest_run_id_from_version,
)


def create_recurring_run_after_upload(
    kfp_client, yaml_path, pipeline_name, version_name, cron_expr
):
    pipeline_id, version_id, _ = get_or_upload_pipeline(
        kfp_client, yaml_path, pipeline_name, version_name
    )

    # create or get experiment
    exp = kfp_client.create_experiment(name=pipeline_name)

    # get latest run of that version in this experiment
    run_id = get_latest_run_id_from_version(kfp_client, exp.id, version_id)

    # schedule job
    return create_recurring_run(kfp_client, run_id, cron_expr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload pipeline YAML and schedule recurring run."
    )
    parser.add_argument(
        "--kfp-api-url", required=True, help="Kubeflow Pipelines API URL"
    )
    parser.add_argument("--kfp-dex-username", required=True, help="Dex username")
    parser.add_argument("--kfp-dex-password", required=True, help="Dex password")
    parser.add_argument(
        "--kfp-dex-auth-type", default="local", help="Dex auth type (default=local)"
    )
    parser.add_argument("--pipeline-name", required=True, help="Pipeline display name")
    parser.add_argument(
        "--version-name", required=True, help="Pipeline version display name"
    )
    parser.add_argument(
        "--cron-expr", default="0 3 * * *", help="Cron expression for recurring job"
    )

    args = parser.parse_args()

    client_auth_manager = KFPClientManager(
        api_url=args.kfp_api_url,
        dex_username=args.kfp_dex_username,
        dex_password=args.kfp_dex_password,
        dex_auth_type=args.kfp_dex_auth_type,
        skip_tls_verify=True,
    )
    kfp_client = client_auth_manager.create_kfp_client()
    print("✅ Authenticated KFP client created.")

    create_recurring_run_after_upload(
        kfp_client,
        yaml_path="kfp_outside/pipeline.yaml",
        pipeline_name=args.pipeline_name,
        version_name=args.version_name,
        cron_expr=args.cron_expr,
    )
