from datetime import datetime
import os
from kfp_outside.utils import KFPClientManager


def main(run_id: str):
    client_auth_manager = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        dex_username=os.getenv("KFP_DEX_USERNAME"),
        dex_password=os.getenv("KFP_DEX_PASSWORD"),
        dex_auth_type=os.getenv("KFP_DEX_AUTH_TYPE", "local"),
        skip_tls_verify=os.getenv("KFP_SKIP_TLS_VERIFY", "false").lower() == "true",
    )
    client = client_auth_manager.create_kfp_client()
    run_detail = client.get_run(run_id)

    print("run_detail fields:", dir(run_detail))
    print("pipeline_version_id:", getattr(run_detail, "pipeline_version_id", None))
    print(
        "pipeline_version_reference:",
        getattr(run_detail, "pipeline_version_reference", None),
    )
    print("pipeline_id:", getattr(run_detail, "pipeline_id", None))
    print("pipeline_spec:", run_detail.pipeline_spec)

    pipeline_version_ref = getattr(run_detail, "pipeline_version_reference", None)
    if pipeline_version_ref:
        print(
            "pipeline_version_reference __dict__:",
            getattr(pipeline_version_ref, "__dict__", pipeline_version_ref),
        )

    # Nếu vẫn None, lấy id từ list_pipelines
    pipelines = client.list_pipelines().pipelines
    for pl in pipelines:
        print("Pipeline:", pl.id, pl.display_name)
        versions = client.list_pipeline_versions(pipeline_id=pl.id).pipeline_versions
        for v in versions:
            print("  Version:", v.id, v.name)

    # Tạm dừng ở đây, bạn sẽ nhìn thấy id thật để điền vào cho recurring run!


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    main(args.run_id)
