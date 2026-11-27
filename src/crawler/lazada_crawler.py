import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.crawler.base_crawler import BaseCrawler
from src.utils.logger import get_logger


logger = get_logger("lazada_crawler")


class LazadaCrawler(BaseCrawler):
    """Collect public product reviews from Lazada Vietnam."""

    REVIEW_URL = "https://my.lazada.vn/pdp/review/getReviewList"
    PAGE_SIZE = 20
    PAGE_DELAY_SECONDS = 1.0

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ):
        super().__init__(platform_name="lazada")
        self.session = session or requests.Session()
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.lazada.vn/",
        }
        self.session.headers.update(self.headers)

    @staticmethod
    def _parse_payload(payload: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            raise ValueError("Lazada review API trả về JSON không phải object.")
        model = payload.get("model")
        if not isinstance(model, dict):
            raise ValueError("Lazada review API không có object 'model'.")
        items = model.get("items", [])
        if not isinstance(items, list):
            raise ValueError("Trường 'model.items' của Lazada không phải danh sách.")
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("Danh sách review Lazada chứa bản ghi không hợp lệ.")
        return model, items

    @staticmethod
    def _created_at(value: Any) -> str:
        if value is None or value == "":
            return ""
        if isinstance(value, (int, float)):
            timestamp = float(value)
            if timestamp > 10_000_000_000:
                timestamp /= 1000
            try:
                return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except (OverflowError, OSError, ValueError):
                return ""
        return str(value).strip()

    @staticmethod
    def _total_pages(model: Dict[str, Any]) -> Optional[int]:
        paging = model.get("paging")
        if not isinstance(paging, dict):
            return None
        try:
            total_pages = int(paging.get("totalPages"))
        except (TypeError, ValueError):
            return None
        return total_pages if total_pages > 0 else None

    def fetch_product_reviews(
        self,
        shop_id: str,
        product_id: str,
        product_name: Optional[str] = None,
        max_reviews: int = 100,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not str(product_id).strip():
            raise ValueError("Lazada cần có product_id.")

        self.reset()
        product_id = str(product_id).strip()
        resolved_shop_id = str(shop_id or "").strip()
        if not resolved_shop_id or resolved_shop_id.lower() == "auto":
            resolved_shop_id = "unknown"
        product_name = product_name or f"Sản phẩm Lazada {product_id}"
        category = category or "Chưa phân loại"
        self.product_info = {
            "platform": self.platform_name,
            "shop_id": resolved_shop_id,
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
        }

        page = 1
        collected = 0
        previous_page_ids = None

        while max_reviews <= 0 or collected < max_reviews:
            response = self.session.get(
                self.REVIEW_URL,
                params={
                    "itemId": product_id,
                    "pageSize": self.PAGE_SIZE,
                    "filter": 0,
                    "sort": 0,
                    "pageNo": page,
                },
                timeout=15,
            )
            response.raise_for_status()
            model, items = self._parse_payload(response.json())
            if not items:
                break

            page_ids = tuple(
                str(item.get("reviewId") or item.get("id") or "")
                for item in items
            )
            if page_ids and page_ids == previous_page_ids:
                logger.warning(
                    "Lazada lặp lại trang review %s cho sản phẩm %s; dừng phân trang.",
                    page,
                    product_id,
                )
                break
            previous_page_ids = page_ids

            for item in items:
                review_id = item.get("reviewId") or item.get("id")
                if review_id is None or not str(review_id).strip():
                    logger.warning(
                        "Bỏ qua review Lazada không có review_id của sản phẩm %s.",
                        product_id,
                    )
                    continue
                added = self.validate_and_add_review(
                    review_id=str(review_id),
                    shop_id=resolved_shop_id,
                    product_id=product_id,
                    product_name=product_name,
                    category=category,
                    rating=item.get("rating"),
                    comment_text=item.get("reviewContent") or "",
                    created_at=self._created_at(item.get("reviewTime")),
                    reviewer_name=item.get("buyerName") or "Khách hàng Lazada",
                    like_count=item.get("helpfulCount", 0),
                    raw_payload=item,
                )
                if added:
                    collected += 1
                    if max_reviews > 0 and collected >= max_reviews:
                        break

            total_pages = self._total_pages(model)
            if (total_pages is not None and page >= total_pages) or (
                total_pages is None and len(items) < self.PAGE_SIZE
            ):
                break
            page += 1
            time.sleep(self.PAGE_DELAY_SECONDS)

        logger.info(
            "Hoàn thành cào %s review Lazada cho sản phẩm %s.",
            collected,
            product_id,
        )
        return self.reviews
