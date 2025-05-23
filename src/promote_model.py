import os
import sys
import mlflow


def promote(model, from_stage, to_stage, tracking_uri):
    client = mlflow.tracking.MlflowClient(tracking_uri)
    stages = None if from_stage.lower() in ("none", "") else [from_stage.capitalize()]
    versions = client.get_latest_versions(model, stages=stages)
    if not versions:
        print(f"[WARN] No version found in stage '{from_stage}'. Nothing to promote.")
        sys.exit(1)
    v = versions[0].version
    client.transition_model_version_stage(model, v, to_stage.capitalize())
    print(f"[INFO] {model}: v{v}  {from_stage} ➜ {to_stage}")


if __name__ == "__main__":
    model = os.getenv("MODEL_NAME")
    from_stage = os.getenv("FROM_STAGE", "none")
    to_stage = os.getenv("TO_STAGE")
    tracking_uri = os.getenv("TRACKING_URI", "http://mlflow.ducdh.com")

    if not model:
        print("[ERROR] MODEL_NAME environment variable is required")
        sys.exit(1)
    if not to_stage:
        print("[ERROR] TO_STAGE environment variable is required")
        sys.exit(1)

    promote(model, from_stage, to_stage, tracking_uri)
