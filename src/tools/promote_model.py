import argparse
import mlflow


def promote(model, from_stage, to_stage, tracking_uri):
    client = mlflow.tracking.MlflowClient(tracking_uri)
    stages = None if from_stage.lower() in ("none", "") else [from_stage.capitalize()]
    versions = client.get_latest_versions(model, stages=stages)

    version = versions[0].version
    client.transition_model_version_stage(model, version, to_stage.capitalize())
    print(f"[INFO] {model}: v{version}  {from_stage} ➜ {to_stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Promote an MLflow model version to a new stage."
    )
    parser.add_argument("--model", required=True, help="Model name to promote")
    parser.add_argument(
        "--from-stage",
        default="none",
        help="Current stage (e.g., none, staging, production)",
    )
    parser.add_argument(
        "--to-stage", required=True, help="Target stage (e.g., staging, production)"
    )
    parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI")
    args = parser.parse_args()

    promote(args.model, args.from_stage, args.to_stage, args.tracking_uri)
