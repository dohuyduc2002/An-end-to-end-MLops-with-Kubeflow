# kfp_components/spark_app.py
from kfp.dsl import component


@component(
    base_image="python:3.11",
    packages_to_install=["kubernetes==28.1.0", "pyyaml==6.0.2"],
)
def run_spark_application(yaml_text: str, namespace: str = "kubeflow"):
    import time, yaml
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
    api = client.CustomObjectsApi()
    spec = yaml.safe_load(yaml_text)
    spec.setdefault("metadata", {}).setdefault("namespace", namespace)
    obj = api.create_namespaced_custom_object(
        group="sparkoperator.k8s.io",
        version="v1beta2",
        namespace=namespace,
        plural="sparkapplications",
        body=spec,
    )
    name = obj["metadata"]["name"]
    while True:
        app = api.get_namespaced_custom_object(
            group="sparkoperator.k8s.io",
            version="v1beta2",
            namespace=namespace,
            plural="sparkapplications",
            name=name,
        )
        st = app.get("status", {}).get("applicationState", {}).get("state")
        if st in ("COMPLETED", "FAILED"):
            if st != "COMPLETED":
                raise RuntimeError(f"SparkApplication {name} failed")
            break
        time.sleep(10)
