import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.sql_server_store import SQLServerStore
from src.utils.logger import get_logger

logger = get_logger("export_sql_server_to_csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export review rows from SQL Server to CSV."
    )
    parser.add_argument(
        "--output",
        default="data/exports/reviews_export.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    with SQLServerStore() as store:
        store.ping()
        exported = store.export_reviews_csv(args.output)
    logger.info(f"Exported {exported} reviews to {args.output}.")


if __name__ == "__main__":
    main()
