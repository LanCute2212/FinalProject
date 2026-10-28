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

    def save_raw_data(self, output_dir: str = "data/raw", filename_prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Lưu dữ liệu cào được thành CSV và JSON (BR11).
        Quy tắc:
        - Mỗi nền tảng/sàn lưu cố định vào 1 file duy nhất: <platform>_raw.csv và <platform>_raw.json.
        - Tự động gộp (append) với dữ liệu cũ và loại bỏ trùng lặp theo review_id.
        """
        os.makedirs(output_dir, exist_ok=True)
        if not self.reviews:
            logger.warning("Không có dữ liệu review mới nào để lưu.")
            return {}

        base_name = filename_prefix if filename_prefix else f"{self.platform_name}_raw"
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        json_path = os.path.join(output_dir, f"{base_name}.json")

        new_df = pd.DataFrame(self.reviews)

        # Nếu file đã tồn tại, đọc lên và gộp dữ liệu
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                logger.info(f"Tìm thấy file cũ '{csv_path}' với {len(existing_df)} review. Tiến hành gộp và loại bỏ trùng lặp...")
                merged_df = pd.concat([existing_df, new_df], ignore_index=True)
                if "review_id" in merged_df.columns:
                    merged_df["review_id"] = merged_df["review_id"].astype(str)
                    before_dedup = len(merged_df)
                    merged_df = merged_df.drop_duplicates(subset=["review_id"], keep="last")
                    dedup_count = before_dedup - len(merged_df)
                    if dedup_count > 0:
                        logger.info(f"Đã khử {dedup_count} review bị trùng lặp.")
                final_df = merged_df
            except Exception as e:
                logger.warning(f"Không thể đọc file cũ ({e}), tiến hành ghi đè dữ liệu mới.")
                final_df = new_df
        else:
            final_df = new_df

        # Lưu CSV
        final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Lưu JSON
        json_records = final_df.to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_records, f, ensure_ascii=False, indent=2)

        logger.info(f"Đã đồng bộ tổng cộng {len(final_df)} bản ghi của sàn '{self.platform_name}' vào:")
        logger.info(f" - CSV : {csv_path}")
        logger.info(f" - JSON: {json_path}")

        return {"csv": csv_path, "json": json_path}

    def reset(self) -> None:
        """Xóa dữ liệu phiên hiện tại trước khi bắt đầu một lần cào mới."""
        self.reviews.clear()
