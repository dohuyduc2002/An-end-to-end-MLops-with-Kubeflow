import os
import argparse
import mlflow


def fetch_parent_run_id(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    max_results: int = 10,
) :
    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id

    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time desc"],
        max_results=max_results,
    )

    return runs.iloc[0]["run_id"].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLflow parent run_id.")
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI (env: MLFLOW_TRACKING_URI)",
    )
    parser.add_argument(
        "--experiment",
        dest="experiment_name",
        help="Experiment name (env: EXPERIMENT_NAME)",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        help="Display name (tag mlflow.runName) (env: RUN_NAME)",
    )
    args = parser.parse_args()

    rid = fetch_parent_run_id(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(rid)  # Jenkins sẽ capture stdout

