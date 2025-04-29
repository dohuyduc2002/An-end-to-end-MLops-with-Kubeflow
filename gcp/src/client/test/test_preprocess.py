import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from loguru import logger

# Add src/ to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_preprocess_real(tmp_path):
    # Project root
    project_root = Path(__file__).resolve().parents[3]
    client_root = project_root / "src" / "client"
    data_dir = client_root / "data"
    joblib_dir = client_root / "joblib"

    # Local input CSVs
    local_train = data_dir / "application_train"
    local_test = data_dir / "application_test"

    # Reference files
    reference_train = joblib_dir / "processed_train"
    reference_test = joblib_dir / "processed_test"

    # Pre-trained transformer
    transformer_path = joblib_dir / "transformer_joblib"
    assert transformer_path.exists(), "❌ transformer_joblib not found."

    # Load transformer
    transformer = joblib.load(transformer_path)
    binning = transformer["binning_process"]
    selector = transformer["selector"]

    # ====== Apply preprocessing on test input ======
    df_train = pd.read_csv(local_train)
    df_test = pd.read_csv(local_test)

    # Only keep columns that are in binning
    df_train = df_train[[col for col in df_train.columns if col in binning.variable_names]]
    df_test = df_test[[col for col in df_test.columns if col in binning.variable_names]]

    X_train_binned = binning.transform(df_train)
    X_test_binned = binning.transform(df_test)

    X_train_selected = selector.transform(X_train_binned)
    X_test_selected = selector.transform(X_test_binned)

    # Save temporary outputs
    out_train = tmp_path / "preprocessed_train.csv"
    out_test = tmp_path / "preprocessed_test.csv"
    pd.DataFrame(X_train_selected).to_csv(out_train, index=False)
    pd.DataFrame(X_test_selected).to_csv(out_test, index=False)

    # ====== Load reference ======
    ref_train = pd.read_csv(reference_train)
    ref_test = pd.read_csv(reference_test)

    # ====== Assertions ======
    def compare(df1, df2, name):
        np.testing.assert_array_almost_equal(
            df1.values, df2.values, decimal=5,
            err_msg=f"❌ Difference found in {name}"
        )
        logger.info(f"✅ {name} matched!")

    compare(pd.read_csv(out_train), ref_train, "Processed Train Data")
    compare(pd.read_csv(out_test), ref_test, "Processed Test Data")

    logger.info("✅ All preprocessing outputs matched reference data.")

