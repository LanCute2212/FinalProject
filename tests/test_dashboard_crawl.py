import queue
import unittest
from unittest.mock import MagicMock, patch

import app


class DashboardCrawlerPersistenceTests(unittest.TestCase):
    def test_sql_server_crawl_records_run_and_reviews(self):
        crawler = MagicMock()
        crawler.reviews = [{
            "platform": "tiki",
            "product_id": "p1",
            "review_id": "r1",
            "comment_text": "good",
        }]
        crawler.product_info = {
            "platform": "tiki",
            "product_id": "p1",
            "shop_id": "s1",
            "product_name": "Product",
            "category": "Electronics",
        }
        crawler.save_raw_data.return_value = {"csv": "raw.csv"}
        store = MagicMock()
        store.start_crawl_run.return_value = "run-1"
        store.upsert_reviews.return_value = 1

        with patch("src.crawler.tiki_crawler.TikiCrawler", return_value=crawler):
            with patch("app.SQLServerStore", return_value=store):
                app._crawl_worker(
                    platform="tiki",
                    product_id="p1",
                    shop_id="auto",
                    product_name="Product",
                    category="Electronics",
                    max_reviews=10,
                    output_dir="data/raw",
                    run_preprocess=False,
                    log_queue=queue.Queue(),
                    sql_server="localhost",
                    database_name="customer_voice_analytics",
                )

        store.start_crawl_run.assert_called_once()
        store.upsert_reviews.assert_called_once_with(
            crawler.reviews,
            crawl_run_id="run-1",
        )
        store.finish_crawl_run.assert_called_once_with(
            run_id="run-1",
            status="completed",
            reviews_collected=1,
            reviews_upserted=1,
            product=crawler.product_info,
        )
        store.close.assert_called_once()

    def test_crawler_exception_marks_run_failed(self):
        crawler = MagicMock()
        crawler.product_info = {}
        crawler.fetch_product_reviews.side_effect = RuntimeError("crawl failed")
        store = MagicMock()
        store.start_crawl_run.return_value = "run-2"
        log_queue = queue.Queue()

        with patch("src.crawler.tiki_crawler.TikiCrawler", return_value=crawler):
            with patch("app.SQLServerStore", return_value=store):
                app._crawl_worker(
                    platform="tiki",
                    product_id="p1",
                    shop_id="auto",
                    product_name="Product",
                    category="Electronics",
                    max_reviews=10,
                    output_dir="data/raw",
                    run_preprocess=False,
                    log_queue=log_queue,
                    sql_server="localhost",
                    database_name="customer_voice_analytics",
                )

        store.finish_crawl_run.assert_called_once_with(
            run_id="run-2",
            status="failed",
            reviews_collected=0,
            reviews_upserted=0,
            error="crawl failed",
            product={},
        )
        self.assertIn("__DONE__", list(log_queue.queue))

    def test_preprocessing_skips_console_output_and_syncs_processed_reviews(self):
        crawler = MagicMock()
        crawler.reviews = [{
            "platform": "tiki",
            "product_id": "p1",
            "review_id": "r1",
            "comment_text": "good",
        }]
        crawler.product_info = {
            "platform": "tiki",
            "product_id": "p1",
            "shop_id": "s1",
            "product_name": "Product",
            "category": "Electronics",
        }
        crawler.save_raw_data.return_value = {"csv": "raw.csv"}
        store = MagicMock()
        store.start_crawl_run.return_value = "run-3"
        store.upsert_reviews.return_value = 1
        processed = app.pd.DataFrame([{
            "platform": "tiki",
            "product_id": "p1",
            "review_id": "r1",
            "is_valid_for_analysis": True,
        }])

        with patch("src.crawler.tiki_crawler.TikiCrawler", return_value=crawler):
            with patch("app.SQLServerStore", return_value=store):
                with patch(
                    "scripts.run_preprocess.run_full_preprocessing",
                    return_value=processed,
                ) as run_preprocessing:
                    app._crawl_worker(
                        platform="tiki",
                        product_id="p1",
                        shop_id="auto",
                        product_name="Product",
                        category="Electronics",
                        max_reviews=10,
                        output_dir="data/raw",
                        run_preprocess=True,
                        log_queue=queue.Queue(),
                        sql_server="localhost",
                        database_name="customer_voice_analytics",
                    )

        run_preprocessing.assert_called_once_with(
            raw_dir="data/raw",
            sync_sql_server=False,
            show_statistics=False,
        )
        self.assertEqual(store.upsert_reviews.call_count, 2)
        store.upsert_reviews.assert_any_call(
            crawler.reviews,
            crawl_run_id="run-3",
        )
        store.upsert_reviews.assert_any_call(processed.to_dict("records"))


if __name__ == "__main__":
    unittest.main()
