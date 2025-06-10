import argparse
import sys

import mlflow
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Determine if a new model should be promoted."
    )
    parser.add_argument(
        "--tracking-uri", required=True, help="MLflow tracking URI (http://...)"
    )
    parser.add_argument(
        "--model-name", required=True, help="Registered model name in MLflow"
    )
    parser.add_argument(
        "--stage",
        required=True,
        help="Target stage to check against (e.g. 'staging', 'production')",
    )
    return parser.parse_args()


def need_promote(client: MlflowClient, model_name: str, stage: str):
    """Return True if latest version is *not* yet in `stage` ⇒ need promote."""
    #https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html: refer to the mlflow docs to search for model versions in the Postgres database
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        return True

    stage_lower = stage.lower()

    # Get the latest version of the model
    # The version is stored in Postgres DB 
    latest_version = max(versions, key=lambda version: int(version.version))

    # Find the metadata of the latest version if it already in stage
    latest_in_stage = None
    for version in versions:
        if version.current_stage.lower() == stage_lower:
            if latest_in_stage is None or int(version.version) > int(
                latest_in_stage.version
            ):
                latest_in_stage = version

    # If no version is found in the target stage, we need to promote
    if latest_in_stage is None:
        return True
    return int(latest_version.version) > int(latest_in_stage.version)


def main() -> None:
    args = parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    client = MlflowClient()

    if need_promote(client, args.model_name, args.stage):
        # 0 = need promotion
        print(f"✅ Need promote: newest version not in stage '{args.stage}'.")
        sys.exit(0)
    else:
        # 1 = skip
        print(f"ℹ️  Latest version already in stage '{args.stage}', skip promote.")
        sys.exit(1)


if __name__ == "__main__":
    main()
