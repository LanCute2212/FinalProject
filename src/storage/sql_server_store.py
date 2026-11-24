import json
import math
import os
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd
import pyodbc


DEFAULT_DATABASE_NAME = "customer_voice_analytics"
IDENTITY_FIELDS = {"platform", "product_id", "review_id"}
INTERNAL_REVIEW_FIELDS = {
    "_id",
    "first_seen_at",
    "last_seen_at",
    "first_seen_run_id",
    "last_seen_run_id",
}
REVIEW_COLUMNS = [
    "platform",
    "product_id",
    "review_id",
    "shop_id",
    "product_name",
    "category",
    "rating",
    "comment_text",
    "created_at",
    "reviewer_name",
    "like_count",
    "raw_payload",
    "comment_text_raw",
    "comment_text_clean",
    "comment_text_segmented",
    "word_count",
    "char_count",
    "is_spam",
    "spam_reason",
    "is_duplicate",
    "is_valid_for_analysis",
    "filter_status",
]
EMPTY_REVIEW_COLUMNS = [
    "review_id",
    "shop_id",
    "product_id",
    "product_name",
    "category",
    "rating",
    "comment_text",
    "created_at",
    "reviewer_name",
    "like_count",
    "platform",
    "raw_payload",
    "comment_text_raw",
    "comment_text_clean",
    "comment_text_segmented",
    "word_count",
    "char_count",
    "is_spam",
    "spam_reason",
    "is_duplicate",
    "is_valid_for_analysis",
    "filter_status",
]


def sqlserver_is_configured() -> bool:
    return bool(os.getenv("SQLSERVER_SERVER", "").strip())


def _safe_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if hasattr(value, "item"):
        return _safe_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _identifier(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _normalize_review(record: Mapping[str, Any]) -> Dict[str, Any]:
    review = {
        str(key): value for key, value in record.items()
        if key not in INTERNAL_REVIEW_FIELDS
    }
    platform = _identifier(review.get("platform")).lower()
    product_id = _identifier(review.get("product_id"))
    review_id = _identifier(review.get("review_id"))
    if not platform or not product_id or not review_id:
        raise ValueError("Mỗi review cần có platform, product_id và review_id.")
    if len(platform) > 32 or len(product_id) > 128 or len(review_id) > 128:
        raise ValueError("platform tối đa 32 ký tự; product_id/review_id tối đa 128 ký tự.")
    review["platform"] = platform
    review["product_id"] = product_id
    review["review_id"] = review_id
    return {key: _safe_value(value) for key, value in review.items()}


def _json_text(value: Any) -> Optional[str]:
    value = _safe_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def _as_datetime(value: Any) -> Optional[datetime]:
    value = _safe_value(value)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().astimezone(timezone.utc).replace(tzinfo=None)
    return None


def _quoted_identifier(value: str) -> str:
    return "[" + value.replace("]", "]]") + "]"


class SQLServerStore:
    """Relational persistence for products, individual reviews, and crawl runs."""

    def __init__(
        self,
        server: Optional[str] = None,
        database_name: Optional[str] = None,
        driver: Optional[str] = None,
        connection=None,
        initialize: bool = True,
    ) -> None:
        self.server = (
            server if server is not None
            else os.getenv("SQLSERVER_SERVER") or "localhost"
        )
        self.database_name = (
            database_name if database_name is not None
            else os.getenv("SQLSERVER_DATABASE") or DEFAULT_DATABASE_NAME
        )
        self.driver = (
            driver if driver is not None
            else os.getenv("SQLSERVER_DRIVER") or "ODBC Driver 17 for SQL Server"
        )
        if not self.server.strip():
            raise ValueError("SQLSERVER_SERVER không được để trống.")
        if not self.database_name.strip() or len(self.database_name) > 128:
            raise ValueError("Tên database phải có từ 1 đến 128 ký tự.")
        if not self.driver.strip():
            raise ValueError("SQLSERVER_DRIVER không được để trống.")
        self.connection = connection

        if self.connection is None:
            self._ensure_database()
            self.connection = pyodbc.connect(
                self._connection_string(self.database_name),
                timeout=5,
            )
        if initialize:
            self.ensure_schema()

    def _connection_string(self, database_name: str) -> str:
        def bracket(value: str) -> str:
            return "{" + value.replace("}", "}}") + "}"

        return (
            f"DRIVER={bracket(self.driver)};"
            f"SERVER={bracket(self.server)};"
            f"DATABASE={bracket(database_name)};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )

    def _ensure_database(self) -> None:
        connection = pyodbc.connect(
            self._connection_string("master"),
            timeout=5,
            autocommit=True,
        )
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT DB_ID(?)", self.database_name)
            if cursor.fetchone()[0] is None:
                cursor.execute(f"CREATE DATABASE {_quoted_identifier(self.database_name)}")
        finally:
            connection.close()

    def ensure_schema(self) -> None:
        statements = [
            """
            IF OBJECT_ID(N'dbo.products', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.products (
                    platform NVARCHAR(32) COLLATE Latin1_General_100_BIN2 NOT NULL,
                    product_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
                    shop_id NVARCHAR(128) NULL,
                    product_name NVARCHAR(1000) NULL,
                    category NVARCHAR(500) NULL,
                    metadata_json NVARCHAR(MAX) NULL,
                    created_at DATETIME2(3) NOT NULL
                        CONSTRAINT DF_products_created_at DEFAULT SYSUTCDATETIME(),
                    updated_at DATETIME2(3) NOT NULL
                        CONSTRAINT DF_products_updated_at DEFAULT SYSUTCDATETIME(),
                    CONSTRAINT PK_products PRIMARY KEY (platform, product_id)
                )
            END
            """,
            """
            IF OBJECT_ID(N'dbo.reviews', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.reviews (
                    platform NVARCHAR(32) COLLATE Latin1_General_100_BIN2 NOT NULL,
                    product_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
                    review_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL,
                    shop_id NVARCHAR(128) NULL,
                    product_name NVARCHAR(1000) NULL,
                    category NVARCHAR(500) NULL,
                    rating INT NULL,
                    comment_text NVARCHAR(MAX) NULL,
                    created_at DATETIME2(3) NULL,
                    reviewer_name NVARCHAR(500) NULL,
                    like_count INT NULL,
                    raw_payload NVARCHAR(MAX) NULL,
                    comment_text_raw NVARCHAR(MAX) NULL,
                    comment_text_clean NVARCHAR(MAX) NULL,
                    comment_text_segmented NVARCHAR(MAX) NULL,
                    word_count INT NULL,
                    char_count INT NULL,
                    is_spam BIT NULL,
                    spam_reason NVARCHAR(255) NULL,
                    is_duplicate BIT NULL,
                    is_valid_for_analysis BIT NULL,
                    filter_status NVARCHAR(100) NULL,
                    first_seen_at DATETIME2(3) NOT NULL
                        CONSTRAINT DF_reviews_first_seen_at DEFAULT SYSUTCDATETIME(),
                    last_seen_at DATETIME2(3) NULL,
                    first_seen_run_id NVARCHAR(36) NULL,
                    last_seen_run_id NVARCHAR(36) NULL,
                    extra_fields_json NVARCHAR(MAX) NULL,
                    CONSTRAINT PK_reviews PRIMARY KEY (platform, product_id, review_id),
                    CONSTRAINT FK_reviews_products FOREIGN KEY (platform, product_id)
                        REFERENCES dbo.products (platform, product_id)
                )
            END
            """,
            """
            IF OBJECT_ID(N'dbo.crawl_runs', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.crawl_runs (
                    run_id NVARCHAR(36) NOT NULL PRIMARY KEY,
                    platform NVARCHAR(32) NOT NULL,
                    product_id NVARCHAR(128) NOT NULL,
                    shop_id NVARCHAR(128) NULL,
                    product_name NVARCHAR(1000) NULL,
                    category NVARCHAR(500) NULL,
                    requested_reviews INT NOT NULL,
                    status NVARCHAR(20) NOT NULL,
                    reviews_collected INT NOT NULL DEFAULT 0,
                    reviews_upserted INT NOT NULL DEFAULT 0,
                    started_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
                    completed_at DATETIME2(3) NULL,
                    error NVARCHAR(MAX) NULL
                )
            END
            """,
            """
            IF NOT EXISTS (
                SELECT 1 FROM sys.indexes
                WHERE name = N'IX_reviews_product' AND object_id = OBJECT_ID(N'dbo.reviews')
            )
                CREATE INDEX IX_reviews_product ON dbo.reviews (platform, product_id)
            """,
            """
            IF NOT EXISTS (
                SELECT 1 FROM sys.indexes
                WHERE name = N'IX_crawl_runs_product_started' AND object_id = OBJECT_ID(N'dbo.crawl_runs')
            )
                CREATE INDEX IX_crawl_runs_product_started
                ON dbo.crawl_runs (platform, product_id, started_at)
            """,
        ]
        cursor = self.connection.cursor()
        try:
            for statement in statements:
                cursor.execute(statement)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def ping(self) -> None:
        self.connection.cursor().execute("SELECT 1").fetchone()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        self.close()

    @staticmethod
    def _merge_statement(table: str, columns: list, update_columns: list) -> str:
        source = ", ".join(f"? AS [{column}]" for column in columns)
        identity = ["platform", "product_id"]
        if table == "reviews":
            identity.append("review_id")
        match = " AND ".join(
            f"target.[{column}] = source.[{column}]" for column in identity
        )
        matched = ""
        if update_columns:
            updates = ", ".join(
                f"target.[{column}] = source.[{column}]"
                for column in update_columns
            )
            matched = f"WHEN MATCHED THEN UPDATE SET {updates}"
        insert_columns = ", ".join(f"[{column}]" for column in columns)
        insert_values = ", ".join(f"source.[{column}]" for column in columns)
        return f"""
            MERGE dbo.{table} WITH (HOLDLOCK) AS target
            USING (SELECT {source}) AS source
                ON {match}
            {matched}
            WHEN NOT MATCHED THEN
                INSERT ({insert_columns}) VALUES ({insert_values});
        """

    @staticmethod
    def _product_values(product: Mapping[str, Any]) -> Dict[str, Any]:
        platform = _identifier(product.get("platform")).lower()
        product_id = _identifier(product.get("product_id"))
        if not platform or not product_id:
            raise ValueError("Sản phẩm cần có platform và product_id.")
        if len(platform) > 32 or len(product_id) > 128:
            raise ValueError("platform tối đa 32 ký tự; product_id tối đa 128 ký tự.")
        metadata = {
            key: value for key, value in product.items()
            if key not in {"_id", "platform", "product_id", "shop_id", "product_name", "category"}
        }
        return {
            "platform": platform,
            "product_id": product_id,
            "shop_id": _safe_value(product.get("shop_id")),
            "product_name": _safe_value(product.get("product_name")),
            "category": _safe_value(product.get("category")),
            "metadata_json": _json_text(metadata) if metadata else None,
        }

    def _upsert_product(self, cursor, product: Mapping[str, Any]) -> None:
        values = self._product_values(product)
        columns = list(values)
        update_columns = [column for column in columns if column not in IDENTITY_FIELDS]
        cursor.execute(
            self._merge_statement("products", columns, update_columns),
            tuple(values[column] for column in columns),
        )

    def upsert_product(self, product: Mapping[str, Any]) -> None:
        cursor = self.connection.cursor()
        try:
            self._upsert_product(cursor, product)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def upsert_reviews(
        self,
        records: Iterable[Mapping[str, Any]],
        crawl_run_id: Optional[str] = None,
    ) -> int:
        unique_reviews = {}
        product_documents = {}
        for record in records:
            review = _normalize_review(record)
            identity = (
                review["platform"],
                review["product_id"],
                review["review_id"],
            )
            unique_reviews[identity] = review
            product_identity = identity[:2]
            product = product_documents.setdefault(
                product_identity,
                {"platform": identity[0], "product_id": identity[1]},
            )
            for key in ("shop_id", "product_name", "category"):
                value = review.get(key)
                if value is not None and str(value).strip():
                    product[key] = value

        if not unique_reviews:
            return 0

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cursor = self.connection.cursor()
        try:
            for product in product_documents.values():
                self._upsert_product(cursor, product)

            for review in unique_reviews.values():
                values = {
                    column: review[column]
                    for column in REVIEW_COLUMNS
                    if column in review
                }
                if "created_at" in values:
                    values["created_at"] = _as_datetime(values["created_at"])
                if "raw_payload" in values:
                    values["raw_payload"] = _json_text(values["raw_payload"])
                extra_fields = {
                    key: value for key, value in review.items()
                    if key not in REVIEW_COLUMNS and key not in INTERNAL_REVIEW_FIELDS
                }
                if extra_fields:
                    values["extra_fields_json"] = _json_text(extra_fields)
                if crawl_run_id:
                    values["last_seen_at"] = now
                    values["last_seen_run_id"] = crawl_run_id
                    values["first_seen_run_id"] = crawl_run_id

                columns = list(values)
                update_columns = [
                    column for column in columns
                    if column not in IDENTITY_FIELDS and column != "first_seen_run_id"
                ]
                cursor.execute(
                    self._merge_statement("reviews", columns, update_columns),
                    tuple(values[column] for column in columns),
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return len(unique_reviews)

    def start_crawl_run(
        self,
        platform: str,
        product_id: str,
        shop_id: str,
        product_name: str,
        category: str,
        requested_reviews: int,
    ) -> str:
        run_id = str(uuid4())
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO dbo.crawl_runs (
                    run_id, platform, product_id, shop_id, product_name,
                    category, requested_reviews, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, N'running')
                """,
                run_id,
                _identifier(platform).lower(),
                _identifier(product_id),
                _safe_value(shop_id),
                _safe_value(product_name or ""),
                _safe_value(category or ""),
                int(requested_reviews),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return run_id

    def finish_crawl_run(
        self,
        run_id: str,
        status: str,
        reviews_collected: int,
        reviews_upserted: int,
        error: Optional[str] = None,
        product: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if status not in {"completed", "no_reviews", "failed"}:
            raise ValueError(f"Trạng thái crawl run không hợp lệ: {status}")
        shop_id = product.get("shop_id") if product else None
        product_name = product.get("product_name") if product else None
        category = product.get("category") if product else None
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                UPDATE dbo.crawl_runs
                SET status = ?, reviews_collected = ?, reviews_upserted = ?,
                    completed_at = SYSUTCDATETIME(), error = ?,
                    shop_id = COALESCE(?, shop_id),
                    product_name = COALESCE(?, product_name),
                    category = COALESCE(?, category)
                WHERE run_id = ? AND status = N'running'
                """,
                status,
                int(reviews_collected),
                int(reviews_upserted),
                error,
                _safe_value(shop_id),
                _safe_value(product_name),
                _safe_value(category),
                run_id,
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Không tìm thấy crawl run đang chạy: {run_id}")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def load_reviews_dataframe(self) -> pd.DataFrame:
        columns = REVIEW_COLUMNS + [
            "first_seen_at",
            "last_seen_at",
            "first_seen_run_id",
            "last_seen_run_id",
            "extra_fields_json",
        ]
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT " + ", ".join(f"[{column}]" for column in columns)
            + " FROM dbo.reviews"
        )
        records = []
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            extra_fields = record.pop("extra_fields_json")
            record = {
                key: value for key, value in record.items()
                if key not in INTERNAL_REVIEW_FIELDS
            }
            if extra_fields:
                try:
                    record.update(json.loads(extra_fields))
                except json.JSONDecodeError:
                    record["extra_fields_json"] = extra_fields
            records.append(record)
        return pd.DataFrame(records, columns=None if records else EMPTY_REVIEW_COLUMNS)

    def export_reviews_csv(self, output_path: str) -> int:
        frame = self.load_reviews_dataframe()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False, encoding="utf-8-sig")
        return len(frame)
