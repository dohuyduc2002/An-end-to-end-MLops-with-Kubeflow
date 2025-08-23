import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "application_train.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "curated_train_data.csv"


def create_timestamp(
    df, max_days_back=365, round_to_seconds=True, seed=None
):
    n = len(df)
    df = df.copy()

    rng = np.random.default_rng(seed)

    created_date = datetime.now(timezone.utc) - timedelta(days=365)
    if round_to_seconds:
        created_date = created_date.replace(microsecond=0)

    df["created"] = [created_date.isoformat()] * n  #'2024-08-20T12:34:56+00:00'

    days_offset = rng.integers(0, max_days_back + 1, size=n)
    updated_list = [
        created_date + timedelta(days=int(offset)) for offset in days_offset
    ]
    if round_to_seconds:
        updated_list = [dt.replace(microsecond=0) for dt in updated_list]

    df["updated"] = [dt.isoformat() for dt in updated_list]

    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.lower()

    df_new = create_timestamp(df)
    df_new.to_csv(OUTPUT_PATH, index=False)
