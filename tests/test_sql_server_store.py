import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_preprocess import load_all_raw_data
from src.crawler.base_crawler import BaseCrawler
from src.storage.sql_server_store import REVIEW_COLUMNS, SQLServerStore

STORE_LOAD_COLUMNS = REVIEW_COLUMNS + [
    "first_seen_at",
    "last_seen_at",
    "first_seen_run_id",
    "last_seen_run_id",
    "extra_fields_json",
]


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rowcount = 1

    def execute(self, query, *parameters):
        if len(parameters) == 1 and isinstance(parameters[0], tuple):
            parameters = parameters[0]
        self.connection.calls.append((query, parameters))
        return self

    def fetchone(self):
        return (1,)

    def fetchall(self):
        return self.connection.rows


class FakeConnection:
    def __init__(self):
        self.calls = []
        self.rows = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class SQLServerStoreTests(unittest.TestCase):
    def setUp(self):
        self.connection = FakeConnection()
        self.store = SQLServerStore(
            connection=self.connection,
            initialize=False,
        )

    def test_connection_string_braces_odbc_values(self):
        store = SQLServerStore(
            server="localhost",
            database_name="test}database",
            connection=self.connection,
            initialize=False,
        )

        self.assertEqual(
            store._connection_string(store.database_name),
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER={localhost};"
            "DATABASE={test}}database};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;",
        )

    def test_schema_keeps_identity_columns_case_sensitive(self):
        self.store.ensure_schema()

        table_statements = [
            query for query, _ in self.connection.calls
            if "CREATE TABLE dbo." in query
        ]
        self.assertEqual(len(table_statements), 3)
        review_schema = next(
            query for query in table_statements if "dbo.reviews" in query
        )
        self.assertEqual(
            review_schema.count("COLLATE Latin1_General_100_BIN2"),
            3,
        )

    def test_upsert_identity_includes_product_and_deduplicates_same_review(self):
        reviews = [
            {"platform": "Tiki", "product_id": "p1", "review_id": "r1", "comment_text": "old"},
            {"platform": "tiki", "product_id": "p1", "review_id": "r1", "comment_text": "new"},
            {"platform": "tiki", "product_id": "p2", "review_id": "r1", "comment_text": "other product"},
        ]

        count = self.store.upsert_reviews(reviews)

        self.assertEqual(count, 2)
        review_calls = [
            (query, params) for query, params in self.connection.calls
            if "MERGE dbo.reviews" in query
        ]
        self.assertEqual(len(review_calls), 2)
        self.assertEqual(
            {(params[0], params[1], params[2]) for _, params in review_calls},
            {("tiki", "p1", "r1"), ("tiki", "p2", "r1")},
        )
        self.assertNotIn("last_seen_at", review_calls[0][0])
        self.assertEqual(self.connection.commits, 1)

    def test_crawl_upsert_records_run_tracking_fields(self):
        self.store.upsert_reviews(
            [{"platform": "tiki", "product_id": "p1", "review_id": "r1"}],
            crawl_run_id="run-1",
        )

        query, parameters = next(
            (query, parameters) for query, parameters in self.connection.calls
            if "MERGE dbo.reviews" in query
        )
        self.assertIn("last_seen_run_id", query)
        self.assertIn("first_seen_run_id", query)
        self.assertEqual(parameters[-2:], ("run-1", "run-1"))
        self.assertNotIn("target.[first_seen_run_id]", query)

    def test_upsert_rejects_missing_identifiers(self):
        with self.assertRaisesRegex(ValueError, "platform, product_id và review_id"):
            self.store.upsert_reviews([{
                "platform": "tiki",
                "product_id": float("nan"),
                "review_id": "r1",
            }])
        self.assertEqual(self.connection.calls, [])

    def test_finish_crawl_run_rejects_unknown_status(self):
        with self.assertRaisesRegex(ValueError, "Trạng thái crawl run"):
            self.store.finish_crawl_run(
                run_id="run-1",
                status="done",
                reviews_collected=0,
                reviews_upserted=0,
            )
        self.assertEqual(self.connection.calls, [])

    def test_dataframe_hides_tracking_fields_and_restores_extra_fields(self):
        columns = STORE_LOAD_COLUMNS
        row = {
            "platform": "tiki",
            "product_id": "p1",
            "review_id": "r1",
            "comment_text": "good",
            "extra_fields_json": json.dumps({"custom_field": "custom"}),
            "first_seen_run_id": "run-1",
        }
        self.connection.rows = [tuple(row.get(column) for column in columns)]

        frame = self.store.load_reviews_dataframe()

        self.assertEqual(frame.loc[0, "custom_field"], "custom")
        self.assertNotIn("first_seen_run_id", frame.columns)
        self.assertEqual(frame.loc[0, "review_id"], "r1")

    def test_export_writes_csv_to_requested_path(self):
        columns = STORE_LOAD_COLUMNS
        row = {
            "platform": "tiki",
            "product_id": "p1",
            "review_id": "r1",
            "comment_text": "good",
        }
        self.connection.rows = [tuple(row.get(column) for column in columns)]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "reviews.csv"
            exported = self.store.export_reviews_csv(str(output_path))
            frame = pd.read_csv(output_path)

        self.assertEqual(exported, 1)
        self.assertEqual(frame.loc[0, "review_id"], "r1")

    def test_empty_collection_returns_dashboard_columns(self):
        frame = self.store.load_reviews_dataframe()

        self.assertTrue(frame.empty)
        self.assertIn("platform", frame.columns)
        self.assertIn("rating", frame.columns)


class DummyCrawler(BaseCrawler):
    def fetch_product_reviews(self, shop_id, product_id, max_reviews=100, category=None):
        return []


class ReviewIdentityTests(unittest.TestCase):
    def test_csv_append_keeps_same_review_id_for_different_products(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            crawler = DummyCrawler("tiki")
            crawler.validate_and_add_review(
                "r1", "shop", "p1", "Product 1", 5, "old", "2024-01-01"
            )
            crawler.validate_and_add_review(
                "r1", "shop", "p2", "Product 2", 4, "other product", "2024-01-01"
            )
            crawler.save_raw_data(temp_dir)

            next_crawl = DummyCrawler("tiki")
            next_crawl.validate_and_add_review(
                "r1", "shop", "p1", "Product 1", 3, "updated", "2024-01-02"
            )
            next_crawl.save_raw_data(temp_dir)

            frame = pd.read_csv(Path(temp_dir) / "tiki_raw.csv")

        self.assertEqual(len(frame), 2)
        self.assertEqual(
            frame.set_index("product_id").loc["p1", "comment_text"],
            "updated",
        )

    def test_preprocessing_dedup_keeps_different_products_with_same_review_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pd.DataFrame([
                {"platform": "tiki", "product_id": "p1", "review_id": "r1", "comment_text": "one"},
                {"platform": "tiki", "product_id": "p2", "review_id": "r1", "comment_text": "two"},
            ]).to_csv(Path(temp_dir) / "tiki_raw.csv", index=False)

            frame = load_all_raw_data(temp_dir)

        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["product_id"]), {"p1", "p2"})


if __name__ == "__main__":
    unittest.main()
