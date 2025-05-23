# utils.py
import re
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
        resp = session.get(self._api_url, allow_redirects=True, verify=not self._skip_tls_verify)
        if resp.status_code == 403:
            url_obj = urlsplit(resp.url)._replace(
                path="/oauth2/start", query=urlencode({"rd": urlsplit(resp.url).path})
            )
            resp = session.get(url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify)
        elif resp.status_code != 200:
            raise RuntimeError(f"GET {self._api_url} returned {resp.status_code}")

        if len(resp.history) == 0:
            return ""

        # follow to dex login
        url_obj = urlsplit(resp.url)
        if url_obj.path.endswith("/auth"):
            url_obj = url_obj._replace(path=url_obj.path + f"/{self._dex_auth_type}")
        resp = session.get(url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify)
        if resp.status_code != 200:
            raise RuntimeError(f"GET {url_obj.geturl()} returned {resp.status_code}")
        dex_login_url = resp.url

        # post credentials
        resp = session.post(dex_login_url,
                      data={"login": self._dex_username, "password": self._dex_password},
                      allow_redirects=True, verify=not self._skip_tls_verify)
        if resp.status_code != 200 or len(resp.history) == 0:
            raise RuntimeError("Dex login failed")

        # if approval step
        if resp.url.endswith("/approval"):
            resp = session.post(resp.url, data={"approval": "approve"},
                          allow_redirects=True, verify=not self._skip_tls_verify)
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
