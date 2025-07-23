import pandas as pd
import pytest
import mlflow
import optuna
from src.model_pipeline.scripts.component_utils import (
    get_lists,
    iv_score,
    select_survivors,
    fit_binning,
    fit_selector,
    build_objective,
    run_optuna_study,
)

@pytest.mark.kfp_components_logic
def test_get_lists(fake_csv):
    df = pd.read_csv(fake_csv)
    cat, num = get_lists(df)
    assert isinstance(cat, list)
    assert isinstance(num, list)


@pytest.mark.kfp_components_logic
def test_iv_score():
    bins = [0, 1, 0, 1, 0, 1]
    y = [0, 1, 0, 1, 0, 1]
    score = iv_score(bins, y)
    assert isinstance(score, float)


@pytest.mark.kfp_components_logic
def test_select_survivors(fake_csv):
    df = pd.read_csv(fake_csv)

    cat, num = get_lists(df)
    survivors = select_survivors(
        df_train=df,
        cat_cols=cat,
        num_cols=num,
        iv_min=0,
        iv_max=10,
        missing_thres=0.5,
    )
    assert isinstance(survivors, list)


@pytest.mark.kfp_components_logic
def test_fit_binning(fake_csv):
    df = pd.read_csv(fake_csv)

    cat, num = get_lists(df)
    survivors = (cat + num)[:2]  
    X_train, X_test = df[survivors], df[survivors]
    y = df["TARGET"]
    binning, df_train_binned, df_test_binned = fit_binning(
        X_train, X_test, y, survivors, cat
    )
    assert df_train_binned.shape[0] == X_train.shape[0]
    assert df_test_binned.shape[0] == X_test.shape[0]
    assert "TARGET" in df_train_binned.columns


@pytest.mark.kfp_components_logic
def test_fit_selector(fake_csv):
    df = pd.read_csv(fake_csv)

    cat, num = get_lists(df)
    survivors = (cat + num)[:2]
    X = df[survivors]
    y = df["TARGET"]
    # Dummy binning
    df_train_binned = pd.get_dummies(
        X, columns=[col for col in survivors if col in cat]
    ).fillna(0)
    df_train_binned["TARGET"] = y
    df_test_binned = df_train_binned.copy().drop("TARGET", axis=1)
    selector, out_train, out_test = fit_selector(
        df_train_binned.drop("TARGET", axis=1),
        df_test_binned,
        y,
        n_features_to_select=1,
    )
    assert out_train.shape[1] == 2  # 1 selected + TARGET
    assert out_test.shape[1] == 1


@pytest.mark.kfp_components_logic
def test_build_objective_and_run_optuna(fake_csv):    
    df = pd.read_csv(fake_csv)
    cat, num = get_lists(df)
    survivors = (cat + num)[:4] if len(cat + num) >= 4 else (cat + num)

    # Encode categorical columns to int
    X = df[survivors].copy()
    for col in cat:
        if col in X.columns:
            X[col], _ = pd.factorize(X[col])
    X = X.fillna(0)
    y = df["TARGET"]

    mlflow.set_tracking_uri("sqlite:///mlruns_test.db")
    mlflow.set_experiment("abc")
    objective = build_objective(X, y, model_type="xgb", suffix="")
    best_trial, best_run_id = run_optuna_study(objective, n_trials=1)
    
    assert isinstance(best_trial, optuna.trial.FrozenTrial)