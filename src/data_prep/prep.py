import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent / "data"

TRAIN_PATH = BASE_PATH / "application_train.csv"
TEST_PATH = BASE_PATH / "application_test.csv"

FEATURES_PATH = BASE_PATH / "curated_features.csv"
LABELS_PATH = BASE_PATH / "curated_labels.csv"


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    labels_df = train_df[["SK_ID_CURR", "TARGET"]].copy()
    features_train = train_df.drop(columns=["TARGET"])

    features_all = pd.concat([features_train, test_df], ignore_index=True)

    features_all.to_csv(FEATURES_PATH, index=False)
    labels_df.to_csv(LABELS_PATH, index=False)
