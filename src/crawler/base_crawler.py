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
    - Chuẩn hóa thông tin: Tên sản phẩm, Danh mục (Category), Tên người đánh giá (Reviewer), Lượt thích (Like count).
    """

    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.reviews: List[Dict[str, Any]] = []
        self.product_info: Dict[str, Any] = {}

    @abstractmethod
    def fetch_product_reviews(
        self,
        shop_id: str,
        product_id: str,
        max_reviews: int = 100,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
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
        category: str = "Chưa phân loại",
        reviewer_name: str = "Khách hàng",
        like_count: int = 0,
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

        # Chuẩn hóa số lượt like
        try:
            like_count_val = int(like_count) if like_count is not None else 0
        except (ValueError, TypeError):
            like_count_val = 0

        record = {
            "review_id": str(review_id),
            "shop_id": str(shop_id),
            "product_id": str(product_id),
            "product_name": str(product_name).strip(),
            "category": str(category).strip() if category else "Chưa phân loại",
            "rating": rating,
            "comment_text": comment_text,
            "created_at": str(created_at),
            "reviewer_name": str(reviewer_name).strip() if reviewer_name else "Khách hàng",
            "like_count": max(0, like_count_val),
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
        - Tự động gộp (append) với dữ liệu cũ và loại trùng theo platform, product_id, review_id.
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

                # Bổ sung các cột mới nếu file cũ chưa có
                for col, default_val in [("category", "Chưa phân loại"), ("reviewer_name", "Khách hàng"), ("like_count", 0)]:
                    if col not in existing_df.columns:
                        existing_df[col] = default_val

                merged_df = pd.concat([existing_df, new_df], ignore_index=True)
                if "platform" not in merged_df.columns:
                    merged_df["platform"] = self.platform_name
                if "review_id" in merged_df.columns:
                    merged_df["review_id"] = merged_df["review_id"].astype(str)
                    before_dedup = len(merged_df)
                    identity_columns = [
                        column for column in ("platform", "product_id", "review_id")
                        if column in merged_df.columns
                    ]
                    merged_df = merged_df.drop_duplicates(
                        subset=identity_columns or ["review_id"],
                        keep="last",
                    )
                    dedup_count = before_dedup - len(merged_df)
                    if dedup_count > 0:
                        logger.info(f"Đã khử {dedup_count} review bị trùng lặp.")
                final_df = merged_df
            except Exception as e:
                logger.warning(f"Không thể đọc file cũ ({e}), tiến hành ghi đè dữ liệu mới.")
                final_df = new_df
        else:
            final_df = new_df

        # Sắp xếp các cột theo thứ tự chuẩn hóa
        desired_columns = [
            "review_id", "shop_id", "product_id", "product_name", "category",
            "rating", "comment_text", "created_at", "reviewer_name", "like_count",
            "platform", "raw_payload"
        ]
        cols_to_use = [c for c in desired_columns if c in final_df.columns] + [c for c in final_df.columns if c not in desired_columns]
        final_df = final_df[cols_to_use]

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
        self.product_info.clear()
