import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.sql_server_store import SQLServerStore
from src.utils.logger import get_logger

logger = get_logger("import_csv_to_sql_server")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import reviews from a CSV file into SQL Server."
    )
    parser.add_argument(
        "--input",
        default="data/processed/reviews_processed.csv",
        help="Input CSV (defaults to the processed review dataset).",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    with SQLServerStore() as store:
        store.ping()
        imported = store.upsert_reviews(frame.to_dict("records"))
    logger.info(f"Imported/upserted {imported} reviews from {args.input}.")


if __name__ == "__main__":
    main()
