from typing import Dict, Optional
from urllib.parse import urlsplit, urlencode
import kfp
import requests
import urllib3

SCIPY_IMAGE = "microwave1005/scipy-img:latest"


class KFPClientManager:
    """
    Class to create a kfp.Client authenticated via Dex.
    """

    def __init__(
        self,
        api_url: str,
        dex_username: str,
        dex_password: str,
        dex_auth_type: str = "local",
        skip_tls_verify: bool = False,
    ):
        self._api_url = api_url
        self._skip_tls_verify = skip_tls_verify
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type

        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if self._dex_auth_type not in ["ldap", "local"]:
            raise ValueError(
                f"Invalid `dex_auth_type` '{self._dex_auth_type}', must be one of ['ldap','local']"
            )

    def _get_session_cookies(self) -> str:
        session = requests.Session()
        resp = session.get(
            self._api_url, allow_redirects=True, verify=not self._skip_tls_verify
        )
        if resp.status_code == 403:
            url_obj = urlsplit(resp.url)._replace(
                path="/oauth2/start", query=urlencode({"rd": urlsplit(resp.url).path})
            )
            resp = session.get(
                url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify
            )
        elif resp.status_code != 200:
            raise RuntimeError(f"GET {self._api_url} returned {resp.status_code}")

        if len(resp.history) == 0:
            return ""

        # follow to dex login
        url_obj = urlsplit(resp.url)
        if url_obj.path.endswith("/auth"):
            url_obj = url_obj._replace(path=url_obj.path + f"/{self._dex_auth_type}")
        resp = session.get(
            url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify
        )
        if resp.status_code != 200:
            raise RuntimeError(f"GET {url_obj.geturl()} returned {resp.status_code}")
        dex_login_url = resp.url

        # post credentials
        resp = session.post(
            dex_login_url,
            data={"login": self._dex_username, "password": self._dex_password},
            allow_redirects=True,
            verify=not self._skip_tls_verify,
        )
        if resp.status_code != 200 or len(resp.history) == 0:
            raise RuntimeError("Dex login failed")

        # if approval step
        if resp.url.endswith("/approval"):
            resp = session.post(
                resp.url,
                data={"approval": "approve"},
                allow_redirects=True,
                verify=not self._skip_tls_verify,
            )
            if resp.status_code != 200:
                raise RuntimeError("Dex approval failed")

        return "; ".join(f"{c.name}={c.value}" for c in session.cookies)

    def _create_kfp_client(self) -> kfp.Client:
        cookies = self._get_session_cookies()

        original = kfp.Client._load_config

        def patched(self_, *a, **k):
            cfg = original(self_, *a, **k)
            cfg.verify_ssl = not self._skip_tls_verify
            return cfg

        kfp.Client._load_config = patched

        return kfp.Client(host=self._api_url, cookies=cookies)

    def create_kfp_client(self) -> kfp.Client:
        return self._create_kfp_client()


def get_or_upload_pipeline(kfp_client, pipeline_yaml, pipeline_name, version_name):
    pipeline_id = None
    version_id = None

    # Get pipeline id by display_name in the dict
    pipelines_resp = kfp_client.list_pipelines(page_size=1000)
    pipelines = pipelines_resp.pipelines
    for pipeline in pipelines:
        if getattr(pipeline, "display_name") == pipeline_name:
            pipeline_id = getattr(pipeline, "pipeline_id")
            break

    if pipeline_id:
        print(f"✅ Found existing pipeline: {pipeline_name} (id={pipeline_id})")
        # check if version_name exists
        versions_list = kfp_client.list_pipeline_versions(
            pipeline_id=pipeline_id, page_size=100
        )
        versions = versions_list.pipeline_versions
        for version in versions:
            name = getattr(version, "display_name")
            if name == version_name:
                version_id = getattr(version, "pipeline_version_id")

                print(
                    f"✅ Found existing pipeline version: {version_name} (id={version_id})"
                )
                break
        if not version_id:
            # Upload version if not found
            pipeline_version = kfp_client.upload_pipeline_version(
                pipeline_package_path=pipeline_yaml,
                pipeline_version_name=version_name,
                pipeline_id=pipeline_id,
            )
            version_id = getattr(pipeline_version, "pipeline_version_id")
            print(f"⬆️  Uploaded new pipeline version: {version_name} (id={version_id})")
    else:
        # Upload pipeline
        pipeline = kfp_client.upload_pipeline(
            pipeline_package_path=pipeline_yaml,
            pipeline_name=pipeline_name,
            namespace="kubeflow-user-example-com",
        )
        pipeline_id = getattr(pipeline, "pipeline_id")
        print(f"⬆️  Uploaded pipeline: {pipeline_name} (id={pipeline_id})")
        pipeline_version = kfp_client.upload_pipeline_version(
            pipeline_package_path=pipeline_yaml,
            pipeline_version_name=version_name,
            pipeline_id=pipeline_id,
        )
        version_id = getattr(pipeline_version, "pipeline_version_id")

        print(f"⬆️  Uploaded pipeline version: {version_name} (id={version_id})")

    return pipeline_id, version_id, version_name


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


def get_latest_run_id_from_version(kfp_client, experiment_id, version_id):
    """Find latest run in experiment that uses the given version_id."""
    runs = (
        kfp_client.list_runs(
            experiment_id=experiment_id, page_size=100, sort_by="created_at desc"
        ).runs
        or []
    )
    for run in runs:
        ref = getattr(run, "pipeline_version_reference", None)
        if ref and getattr(ref, "pipeline_version_id", None) == version_id:
            return run.run_id
    raise ValueError("❌ No run found for this version in the experiment.")
