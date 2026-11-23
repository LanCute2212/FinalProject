import time
import random
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.crawler.base_crawler import BaseCrawler
from src.utils.logger import get_logger

logger = get_logger("shopee_crawler")

class ShopeeCrawler(BaseCrawler):
    """
    Crawler thu thập review từ Shopee thông qua Endpoint API v2 get_ratings.
    Tuân thủ BR04, BR05, BR06, BR08, BR10, BR11.
    Bổ sung trích xuất: Danh mục (Category), Tên người đánh giá (Reviewer), Số lượt thích (Like count).
    """

    BASE_URL = "https://shopee.vn/api/v2/item/get_ratings"

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        super().__init__(platform_name="shopee")
        self.session = requests.Session()
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": "https://shopee.vn/",
            "Accept": "application/json",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "X-Requested-With": "XMLHttpRequest"
        }
        self.session.headers.update(self.headers)

    def fetch_product_reviews(
        self,
        shop_id: str,
        product_id: str,
        product_name: str = "Sản phẩm Shopee",
        max_reviews: int = 100,
        category: str = "Thiết bị điện tử",
        filter_type: int = 1  # 1: chỉ lấy các review có văn bản (filter with comments)
    ) -> List[Dict[str, Any]]:
        """
        Cào review cho 1 sản phẩm của 1 Shop trên Shopee.
        """
        self.reset()
        logger.info(f"Bắt đầu cào Shopee: Shop ID {shop_id}, Item ID {product_id}, Tên: [{product_name}], Danh mục: [{category}]")
        
        offset = 0
        limit = 20
        collected_for_item = 0

        while collected_for_item < max_reviews:
            params = {
                "itemid": product_id,
                "shopid": shop_id,
                "offset": offset,
                "limit": limit,
                "filter": filter_type,
                "type": 0,
                "flag": 1
            }

            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=12)
                if resp.status_code != 200:
                    logger.warning(f"Shopee API trả về status code {resp.status_code}. Có thể bị rate-limit hoặc chặn WAF.")
                    break

                json_resp = resp.json()
                data = json_resp.get("data", {})
                ratings = data.get("ratings", [])

                if not ratings:
                    logger.info("Không còn review nào tiếp theo.")
                    break

                for r in ratings:
                    review_id = str(r.get("cmtid", f"cm_{r.get('orderid', offset)}"))
                    rating_star = r.get("rating_star", 5)
                    comment = r.get("comment", "")
                    
                    # Trích xuất số lượt thích và tên người đánh giá Shopee
                    like_count = r.get("like_count", 0)
                    reviewer_name = r.get("author_username") or f"Khách hàng Shopee {r.get('userid', '')}".strip()

                    ctime = r.get("ctime")
                    if isinstance(ctime, (int, float)):
                        created_at = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    added = self.validate_and_add_review(
                        review_id=review_id,
                        shop_id=shop_id,
                        product_id=product_id,
                        product_name=product_name,
                        category=category,
                        rating=rating_star,
                        comment_text=comment,
                        created_at=created_at,
                        reviewer_name=reviewer_name,
                        like_count=like_count,
                        raw_payload=r
                    )
                    if added:
                        collected_for_item += 1
                        if max_reviews > 0 and collected_for_item >= max_reviews:
                            break

                offset += limit
                time.sleep(random.uniform(1.0, 2.0))

            except Exception as e:
                logger.error(f"Lỗi khi cào Shopee offset {offset}: {e}")
                break

        logger.info(f"Hoàn thành cào {collected_for_item} review từ Shopee cho sản phẩm {product_id}.")
        return self.reviews
