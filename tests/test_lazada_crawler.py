import unittest

from src.crawler.lazada_crawler import LazadaCrawler


BASE_REVIEW_FIELDS = {
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
}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


class LazadaCrawlerTests(unittest.TestCase):
    def test_maps_lazada_review_into_existing_crawler_record_shape(self):
        session = FakeSession([
            FakeResponse({
                "model": {
                    "items": [{
                        "reviewId": "lzd-review-1",
                        "rating": 4,
                        "reviewContent": "Giao hàng nhanh, sản phẩm ổn.",
                        "reviewTime": "2026-11-25 10:30:00",
                        "buyerName": "Người mua",
                        "helpfulCount": 2,
                    }],
                    "paging": {"totalPages": 1},
                }
            })
        ])
        crawler = LazadaCrawler(session=session)

        reviews = crawler.fetch_product_reviews(
            shop_id="seller-1",
            product_id="item-1",
            product_name="Sản phẩm thử nghiệm",
            max_reviews=10,
            category="Điện tử",
        )

        self.assertEqual(len(reviews), 1)
        self.assertEqual(set(reviews[0]), BASE_REVIEW_FIELDS)
        self.assertEqual(reviews[0]["platform"], "lazada")
        self.assertEqual(reviews[0]["review_id"], "lzd-review-1")
        self.assertEqual(reviews[0]["rating"], 4)
        self.assertEqual(session.calls[0][1]["itemId"], "item-1")

    def test_invalid_rating_is_skipped(self):
        session = FakeSession([
            FakeResponse({
                "model": {
                    "items": [{
                        "reviewId": "lzd-review-invalid",
                        "rating": 0,
                        "reviewContent": "Invalid rating.",
                    }],
                    "paging": {"totalPages": 1},
                }
            })
        ])
        crawler = LazadaCrawler(session=session)

        reviews = crawler.fetch_product_reviews(
            shop_id="seller-1",
            product_id="item-1",
            max_reviews=10,
        )

        self.assertEqual(reviews, [])

    def test_missing_model_in_response_raises_clear_error(self):
        session = FakeSession([FakeResponse({"error": "temporarily unavailable"})])
        crawler = LazadaCrawler(session=session)

        with self.assertRaisesRegex(ValueError, "không có object 'model'"):
            crawler.fetch_product_reviews(
                shop_id="seller-1",
                product_id="item-1",
                max_reviews=10,
            )


if __name__ == "__main__":
    unittest.main()
