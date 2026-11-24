import time
import random
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.crawler.base_crawler import BaseCrawler
from src.utils.logger import get_logger

logger = get_logger("tiki_crawler")

class TikiCrawler(BaseCrawler):
    """
    Crawler thu thập review từ sàn Tiki thông qua public API.
    Hỗ trợ cào đa sản phẩm của 1 Seller.
    Bổ sung trích xuất: Danh mục (Category), Tên người đánh giá (Reviewer), Số lượt thích (Like/Thank count).
    """

    BASE_REVIEW_URL = "https://tiki.vn/api/v2/reviews"
    BASE_PRODUCT_URL = "https://tiki.vn/api/v2/products"

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        super().__init__(platform_name="tiki")
        self.session = requests.Session()
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        self.session.headers.update(self.headers)

    def get_product_info(self, product_id: str) -> Dict[str, Any]:
        """Lấy thông tin cơ bản của sản phẩm (tên, danh mục, seller_id)."""
        url = f"{self.BASE_PRODUCT_URL}/{product_id}"
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                
                # Trích xuất danh mục sản phẩm từ breadcrumbs hoặc categories
                category = "Chưa phân loại"
                breadcrumbs = data.get("breadcrumbs", [])
                if breadcrumbs and isinstance(breadcrumbs, list):
                    names = [b.get("name") for b in breadcrumbs if isinstance(b, dict) and b.get("name")]
                    if names:
                        category = " > ".join(names[-2:]) if len(names) >= 2 else names[0]
                elif data.get("categories", {}).get("name"):
                    category = str(data["categories"]["name"])

                return {
                    "product_id": str(product_id),
                    "product_name": data.get("name", f"Sản phẩm {product_id}"),
                    "category": category,
                    "seller_id": str(data.get("current_seller", {}).get("id", "default_seller"))
                }
        except Exception as e:
            logger.warning(f"Lỗi lấy thông tin sản phẩm {product_id}: {e}")
            
        return {
            "product_id": str(product_id),
            "product_name": f"Sản phẩm {product_id}",
            "category": "Chưa phân loại",
            "seller_id": "default_seller"
        }

    def fetch_product_reviews(
        self,
        shop_id: str,
        product_id: str,
        product_name: Optional[str] = None,
        max_reviews: int = 100,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Cào review cho một sản phẩm theo phân trang.
        Nếu max_reviews <= 0: cào TOÀN BỘ review hiện có trên sàn.
        """
        self.reset()
        info = self.get_product_info(product_id)
        if not product_name:
            product_name = info["product_name"]
        if not category:
            category = info.get("category", "Chưa phân loại")
        if shop_id == "auto":
            shop_id = info["seller_id"]
        self.product_info = {
            **info,
            "platform": self.platform_name,
            "shop_id": shop_id,
            "product_name": product_name,
            "category": category,
        }

        limit_desc = f"tối đa {max_reviews}" if max_reviews > 0 else "TOÀN BỘ (không giới hạn)"
        logger.info(f"Bắt đầu cào review Tiki: [{product_name}] (ID: {product_id}, Ngành hàng: [{category}], Giới hạn: {limit_desc})")
        
        page = 1
        limit = 20
        collected_for_item = 0

        while True:
            if max_reviews > 0 and collected_for_item >= max_reviews:
                break

            params = {
                "product_id": product_id,
                "include": "comments",
                "page": page,
                "limit": limit
            }

            try:
                resp = self.session.get(self.BASE_REVIEW_URL, params=params, timeout=10)
                if resp.status_code != 200:
                    logger.warning(f"Tiki API trả về mã lỗi {resp.status_code} tại page {page}.")
                    break

                json_data = resp.json()
                data_list = json_data.get("data", [])
                paging = json_data.get("paging", {})
                total_on_tiki = paging.get("total", 0)

                if page == 1 and total_on_tiki:
                    logger.info(f"Phát hiện sản phẩm có tổng cộng {total_on_tiki} review trên sàn Tiki.")

                if not data_list:
                    logger.info(f"Đã duyệt hết toàn bộ review tại page {page}.")
                    break

                for item in data_list:
                    review_id = str(item.get("id"))
                    rating = item.get("rating", 5)
                    content = item.get("content", "").strip()
                    
                    # Trích xuất số lượt thích / cảm ơn (BR yêu cầu thu thập like_count)
                    like_count = item.get("thank_count", 0)

                    # Trích xuất tên người đánh giá (hoặc mã định danh)
                    created_by = item.get("created_by") or {}
                    reviewer_name = (
                        created_by.get("name") 
                        or created_by.get("full_name") 
                        or (f"Khách hàng Tiki {created_by.get('id')}" if created_by.get("id") else "Khách hàng Tiki")
                    )

                    # Chuyển đổi timestamp tạo review
                    created_at_raw = item.get("created_at")
                    if isinstance(created_at_raw, (int, float)):
                        created_at_str = datetime.fromtimestamp(created_at_raw).strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        created_at_str = str(created_at_raw or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                    added = self.validate_and_add_review(
                        review_id=review_id,
                        shop_id=shop_id,
                        product_id=product_id,
                        product_name=product_name,
                        category=category,
                        rating=rating,
                        comment_text=content,
                        created_at=created_at_str,
                        reviewer_name=reviewer_name,
                        like_count=like_count,
                        raw_payload=item
                    )
                    if added:
                        collected_for_item += 1
                        if max_reviews > 0 and collected_for_item >= max_reviews:
                            break

                logger.info(f"Page {page}: Đã thu thập {collected_for_item} reviews...")
                page += 1
                time.sleep(random.uniform(0.3, 0.7))  # Delay nhỏ tránh rate-limit

            except Exception as e:
                logger.error(f"Lỗi kết nối khi cào Tiki page {page}: {e}")
                break

        logger.info(f"Hoàn thành cào {collected_for_item} review cho sản phẩm {product_id}.")
        return self.reviews
