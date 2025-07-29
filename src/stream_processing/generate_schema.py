import json
from pathlib import Path
import os

from avro_schema import (pos_cash_balance_schema,
                         credit_card_balance_schema,
                         installments_payments_schema,
                         bureau_schema,
                         previous_application_schema,
                         bureau_balance_schema,
                         application_schema)

def main():
    out_dir = "avro"
    os.makedirs(out_dir, exist_ok=True)

    schema_map = {
        "pos_cash_balance_schema.avsc": pos_cash_balance_schema,
        "credit_card_balance_schema.avsc": credit_card_balance_schema,
        "installments_payments_schema.avsc": installments_payments_schema,
        "bureau_schema.avsc": bureau_schema,
        "previous_application_schema.avsc": previous_application_schema,
        "bureau_balance_schema.avsc": bureau_balance_schema,
        "application_schema.avsc": application_schema,
    }

    for filename, schema in schema_map.items():
        out_file = Path(out_dir) / filename
        with open(out_file, "w") as f:
            json.dump(schema, f, indent=2)


if __name__ == "__main__":
    main()
