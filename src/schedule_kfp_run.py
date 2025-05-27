import argparse
from kfp_outside.utils import KFPClientManager


def get_run_params(kfp_client, run_id):
    run = kfp_client.get_run(run_id)
    pipeline_version_reference = getattr(run, "pipeline_version_reference", None)

    pipeline_id = getattr(pipeline_version_reference, "pipeline_id", None)
    pipeline_version_id = getattr(
        pipeline_version_reference, "pipeline_version_id", None
    )
    params = getattr(run.runtime_config, "parameters", None)
    return {
        "experiment_id": run.experiment_id,
        "pipeline_id": pipeline_id,
        "pipeline_version_id": pipeline_version_id,
        "params": params,
    }


def create_recurring_run(kfp_client, run_id, cron_expr):
    run_info = get_run_params(kfp_client, run_id)
    job_name = f"Recurring Job from {run_id}"

    job = kfp_client.create_recurring_run(
        experiment_id=run_info["experiment_id"],
        job_name=job_name,
        description=f"Recurring run for {job_name}",
        cron_expression=cron_expr,
        pipeline_id=run_info["pipeline_id"],
        version_id=run_info["pipeline_version_id"],
        params=run_info["params"],
        enabled=True,
        no_catchup=True,
    )
    print("⏰ Recurring run created:", job)
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule a recurring KFP run from an existing run_id."
    )
    parser.add_argument(
        "--kfp-api-url", required=True, help="Kubeflow Pipelines API URL"
    )
    parser.add_argument(
        "--kfp-dex-username", required=True, help="Dex username for KFP authentication"
    )
    parser.add_argument(
        "--kfp-dex-password", required=True, help="Dex password for KFP authentication"
    )
    parser.add_argument(
        "--kfp-dex-auth-type",
        default="local",
        help="Dex authentication type (default: local)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Base KFP run_id to schedule recurring run from.",
    )
    parser.add_argument(
        "--cron-expr",
        default="0 3 * * *",
        help="Cron expression for recurring run schedule.",
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

    create_recurring_run(
        kfp_client=kfp_client,
        run_id=args.run_id,
        cron_expr=args.cron_expr,
        
    )
