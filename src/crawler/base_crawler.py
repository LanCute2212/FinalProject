import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from src.utils.logger import get_logger

logger = get_logger("base_crawler")

class BaseCrawler(ABC):
    """
    Lớp cơ sở cho các crawler dữ liệu đánh giá sàn TMĐT.
    Tuân thủ các quy tắc nghiệp vụ:
    - BR04: Sản phẩm thuộc về một Seller.
    - BR05: Một Seller có thể có nhiều sản phẩm.
    - BR06: Review phải gắn với sản phẩm cụ thể.
    - BR08: Review thuộc phạm vi Seller.
    - BR10: Rating nằm trong thang 1 - 5.
    - BR11: Bảo toàn review gốc không thay đổi.
    """

    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.reviews: List[Dict[str, Any]] = []

    @abstractmethod
    def fetch_product_reviews(self, shop_id: str, product_id: str, max_reviews: int = 100) -> List[Dict[str, Any]]:
        """Hàm trừu tượng cần được override bởi từng sàn (Shopee, Tiki...)."""
        pass

    def validate_and_add_review(
        self,
        review_id: str,
        shop_id: str,
        product_id: str,
        product_name: str,
        rating: int,
        comment_text: str,
        created_at: str,
        raw_payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Kiểm tra và chuẩn hóa bản ghi review trước khi lưu trữ."""
        # Kiểm tra thang điểm rating (BR10)
        try:
            rating = int(rating)
            if not (1 <= rating <= 5):
                logger.warning(f"Bỏ qua review {review_id}: rating {rating} ngoài phạm vi 1-5.")
                return False
        except (ValueError, TypeError):
            logger.warning(f"Bỏ qua review {review_id}: rating không hợp lệ ({rating}).")
            return False

        # Đảm bảo nội dung văn bản là chuỗi (BR11 bảo toàn gốc)
        if comment_text is None:
            comment_text = ""
        else:
            comment_text = str(comment_text)

        record = {
            "review_id": str(review_id),
            "shop_id": str(shop_id),
            "product_id": str(product_id),
            "product_name": str(product_name).strip(),
            "rating": rating,
            "comment_text": comment_text,
            "created_at": str(created_at),
            "platform": self.platform_name,
            "raw_payload": json.dumps(raw_payload, ensure_ascii=False) if raw_payload else "{}"
        }

        self.reviews.append(record)
        return True

    def save_raw_data(self, output_dir: str = "data/raw", filename_prefix: str = "reviews") -> Dict[str, str]:
        """Lưu toàn bộ dữ liệu cào được thành 2 định dạng: CSV và JSON (bảo toàn raw data BR11)."""
        os.makedirs(output_dir, exist_ok=True)
        if not self.reviews:
            logger.warning("Không có dữ liệu review nào để lưu.")
            return {}

        csv_path = os.path.join(output_dir, f"{filename_prefix}_{self.platform_name}_raw.csv")
        json_path = os.path.join(output_dir, f"{filename_prefix}_{self.platform_name}_raw.json")

        # Lưu CSV
        df = pd.DataFrame(self.reviews)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Lưu JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.reviews, f, ensure_ascii=False, indent=2)

        logger.info(f"Đã lưu {len(self.reviews)} bản ghi thô vào:")
        logger.info(f" - CSV : {csv_path}")
        logger.info(f" - JSON: {json_path}")

        return {"csv": csv_path, "json": json_path}
