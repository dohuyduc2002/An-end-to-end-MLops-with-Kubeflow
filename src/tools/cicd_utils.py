from urllib.parse import urlsplit, urlencode
import kfp
import requests
import urllib3
import logging

SCIPY_IMAGE = "microwave1005/scipy-img:latest"


# Refer to Kubeflow example in setting up KFPClientManager: https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/connect-api/
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


def get_runs_reponse(kfp_client, namespace):
    # We utilize the list_runs API to get the latest run in the specified namespace
    # The list_runs will return a list of runs in a JSON format, which we can parse to get the latest run
    runs = kfp_client.list_runs(
        page_size=10,
        sort_by="namespace desc", # You can use SQL query to sort the runs
        namespace=namespace,
    ).runs

    latest_run = runs[0]
    run_id = latest_run.run_id

    run = kfp_client.get_run(run_id)
    # the object V2beta1Run which is the return of run in python SDK
    # In the JSON response, the run object has a field called "pipeline_version_reference", 
    # which has the same attributes in object V2beta1Run
    pipeline_version_reference =  run.pipeline_version_reference

    pipeline_id = pipeline_version_reference.pipeline_id
    pipeline_version_id =  pipeline_version_reference.pipeline_version_id

    return {
        "experiment_id": run.experiment_id,
        "pipeline_id": pipeline_id,
        "pipeline_version_id": pipeline_version_id,
        "run_id": run_id,
    }


def create_recurring_run_with_params(kfp_client, cron_expr, run_info, params):
    job_name = f"Recurring Job from {run_info['run_id']}"

    job = kfp_client.create_recurring_run(
        experiment_id=run_info["experiment_id"],
        job_name=job_name,
        description=f"Recurring run for {job_name}",
        cron_expression=cron_expr, # THE CRON EXPRESSION HERE IS USING GO EXPRESSION FORMAT
        # https://pkg.go.dev/github.com/robfig/cron#hdr-CRON_Expression_Format
        pipeline_id=run_info["pipeline_id"],
        version_id=run_info["pipeline_version_id"],
        params=params,
        enabled=True,
        no_catchup=True,
    )
    logging.info(f"Created recurring run: {job_name} (id={job.id})")