from kfp import dsl
from component_utils import BASE_IMAGE, TARGET_IMAGE

@dsl.component(base_image=BASE_IMAGE, target_image=TARGET_IMAGE)
def apply_rbac_spark(
    namespace: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    yaml_object: str,
):
    import io, yaml
    from kubernetes import client, config
    from minio import Minio

    config.load_incluster_config()
    core = client.CoreV1Api()
    rbac = client.RbacAuthorizationV1Api()
    mc = Minio(minio_endpoint, minio_access_key, minio_secret_key, secure=False)

    data = mc.get_object(minio_bucket, yaml_object).read()
    docs = [d for d in yaml.safe_load_all(io.BytesIO(data))]

    for obj in docs:
        if "metadata" in obj:
            obj["metadata"]["namespace"] = namespace

        kind = obj["kind"].lower()
        name = obj["metadata"]["name"]

        if kind == "serviceaccount":
            core.replace_namespaced_service_account(name, namespace, obj)
        elif kind == "role":
            rbac.replace_namespaced_role(name, namespace, obj)
        elif kind == "rolebinding":
            rbac.replace_namespaced_role_binding(name, namespace, obj)
        elif kind == "clusterrole":
            rbac.replace_cluster_role(name, obj)
        elif kind == "clusterrolebinding":
            rbac.replace_cluster_role_binding(name, obj)
