import re
from urllib.parse import urlsplit, urlencode
import kfp
import requests
import urllib3

class KFPClientManager:
    def __init__(self, api_url: str, dex_username: str, dex_password: str,
                 dex_auth_type: str = "local", skip_tls_verify: bool = False):
        self._api_url = api_url
        self._skip_tls_verify = skip_tls_verify
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type

        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if self._dex_auth_type not in ["ldap", "local"]:
            raise ValueError(
                f"Invalid `dex_auth_type` '{self._dex_auth_type}', must be one of: ['ldap', 'local']"
            )

    def _get_session_cookies(self) -> str:
        s = requests.Session()
        resp = s.get(self._api_url, allow_redirects=True, verify=not self._skip_tls_verify)
        if resp.status_code == 200:
            pass
        elif resp.status_code == 403:
            url_obj = urlsplit(resp.url)
            url_obj = url_obj._replace(path="/oauth2/start", query=urlencode({"rd": url_obj.path}))
            resp = s.get(url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify)
        else:
            raise RuntimeError(f"HTTP status code '{resp.status_code}' for GET against: {self._api_url}")

        if len(resp.history) == 0:
            return ""

        url_obj = urlsplit(resp.url)
        if re.search(r"/auth$", url_obj.path):
            url_obj = url_obj._replace(path=re.sub(r"/auth$", f"/auth/{self._dex_auth_type}", url_obj.path))
        if re.search(r"/auth/.*/login$", url_obj.path):
            dex_login_url = url_obj.geturl()
        else:
            resp = s.get(url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP status code '{resp.status_code}' for GET against: {url_obj.geturl()}")
            dex_login_url = resp.url

        resp = s.post(dex_login_url, data={"login": self._dex_username, "password": self._dex_password},
                      allow_redirects=True, verify=not self._skip_tls_verify)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP status code '{resp.status_code}' for POST against: {dex_login_url}")
        if len(resp.history) == 0:
            raise RuntimeError(f"Login credentials are probably invalid - no redirect after POST to: {dex_login_url}")

        url_obj = urlsplit(resp.url)
        if re.search(r"/approval$", url_obj.path):
            dex_approval_url = url_obj.geturl()
            resp = s.post(dex_approval_url, data={"approval": "approve"},
                          allow_redirects=True, verify=not self._skip_tls_verify)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP status code '{resp.status_code}' for POST against: {url_obj.geturl()}")
        return "; ".join([f"{c.name}={c.value}" for c in s.cookies])

    def _create_kfp_client(self) -> kfp.Client:
        try:
            session_cookies = self._get_session_cookies()
        except Exception as ex:
            raise RuntimeError("Failed to get Dex session cookies") from ex

        original_load_config = kfp.Client._load_config

        def patched_load_config(client_self, *args, **kwargs):
            config = original_load_config(client_self, *args, **kwargs)
            config.verify_ssl = not self._skip_tls_verify
            return config

        patched_kfp_client = kfp.Client
        patched_kfp_client._load_config = patched_load_config

        return patched_kfp_client(
            host=self._api_url,
            cookies=session_cookies
        )

    def create_kfp_client(self) -> kfp.Client:
        return self._create_kfp_client()
