import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "application_train.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "curated_train_data.csv"

def add_created_updated_columns(df, max_days_back=365):
    n = len(df)

    created_date = datetime.now() - timedelta(days=365)
    created_iso = created_date.isoformat()
    df["created"] = [created_iso] * n

    days_offset = np.random.randint(0, max_days_back + 1, size=n)
    updated_list = [
        created_date + timedelta(days=int(offset)) for offset in days_offset
    ]
    df["updated"] = [dt.isoformat() for dt in updated_list]
    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df_new = add_created_updated_columns(df)
    df_new.to_csv(OUTPUT_PATH, index=False)
