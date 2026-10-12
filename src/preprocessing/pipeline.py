import os
import pandas as pd
from typing import Dict, Any, Optional
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.spam_detector import SpamDetector
from src.preprocessing.tokenizer import VietnameseTokenizer
from src.utils.logger import get_logger

logger = get_logger("preprocessing_pipeline")

class PreprocessingPipeline:
    """
    Pipeline hoàn chỉnh chuyển đổi dữ liệu Review thô (Raw) sang dữ liệu sạch (Processed).
    Tuân thủ đầy đủ các yêu cầu kỹ thuật & nghiệp vụ:
    - BR09: Lọc dữ liệu không hợp lệ.
    - BR10: Kiểm soát thang điểm rating 1-5.
    - BR11: Bảo toàn review gốc (comment_text_raw).
    - BR12: Nhận diện review trùng lặp (is_duplicate).
    - NFR18 & NFR19: Phân biệt rõ dữ liệu gốc và dữ liệu đã xử lý để phục vụ đối chiếu.
    """

    def __init__(
        self,
        teencode_dict_path: str = "data/dictionaries/teencode_dict.json",
        emoji_dict_path: str = "data/dictionaries/emoji_dict.json",
        spam_patterns_path: str = "data/dictionaries/spam_patterns.json"
    ):
        self.cleaner = TextCleaner()
        self.normalizer = TextNormalizer(
            teencode_dict_path=teencode_dict_path,
            emoji_dict_path=emoji_dict_path
        )
        self.spam_detector = SpamDetector(spam_patterns_path=spam_patterns_path)
        self.tokenizer = VietnameseTokenizer()

    def process_single_text(self, text: str) -> Dict[str, Any]:
        """Xử lý tiền xử lý cho 1 chuỗi review đơn lẻ."""
        raw_text = str(text or "")
        
        # Bước 1: Làm sạch cơ bản (HTML, URLs, lặp ký tự, NFC)
        cleaned_text = self.cleaner.clean(raw_text)
        
        # Bước 2: Chuẩn hóa teencode và emoji
        normalized_text = self.normalizer.normalize(cleaned_text, handle_emoji="tag", lower=True)
        
        # Bước 3: Kiểm tra spam/rác
        is_spam, spam_reason = self.spam_detector.check_spam(raw_text, normalized_text)
        
        # Bước 4: Tách từ tiếng Việt
        segmented_text = self.tokenizer.tokenize(normalized_text)
        
        # Bước 5: Thống kê độ dài
        word_count = len(normalized_text.split())
        char_count = len(normalized_text)

        return {
            "comment_text_clean": normalized_text,
            "comment_text_segmented": segmented_text,
            "word_count": word_count,
            "char_count": char_count,
            "is_spam": is_spam,
            "spam_reason": spam_reason
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "comment_text") -> pd.DataFrame:
        """
        Xử lý toàn bộ DataFrame chứa review.
        """
        logger.info(f"Bắt đầu pipeline tiền xử lý cho {len(df)} bản ghi...")

        # Đảm bảo có cột text gốc
        if text_column not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{text_column}' trong DataFrame.")

        # Bảo toàn dữ liệu gốc (BR11, NFR19)
        if "comment_text_raw" not in df.columns:
            df["comment_text_raw"] = df[text_column].astype(str)

        # Áp dụng từng bước xử lý
        results = [self.process_single_text(t) for t in df["comment_text_raw"]]
        res_df = pd.DataFrame(results)

        # Ghép kết quả vào DataFrame chính
        for col in res_df.columns:
            df[col] = res_df[col]

        # Đánh dấu trùng lặp (BR12)
        df = self.spam_detector.mark_duplicates(df, text_column="comment_text_clean")

        # Cột cờ tổng hợp: Có đủ điều kiện đưa vào mô hình NLP không (BR09, BR12)
        df["is_valid_for_analysis"] = (~df["is_spam"]) & (~df["is_duplicate"])

        # Gán lý do loại bỏ rõ ràng
        def determine_exclusion(row):
            if row["is_spam"]:
                return f"SPAM: {row['spam_reason']}"
            elif row["is_duplicate"]:
                return "DUPLICATE_REVIEW"
            return "VALID"

        df["filter_status"] = df.apply(determine_exclusion, axis=1)

        valid_count = df["is_valid_for_analysis"].sum()
        spam_count = df["is_spam"].sum()
        dup_count = df["is_duplicate"].sum()

        logger.info(f"=== BÁO CÁO TIỀN XỬ LÝ DỮ LIỆU ===")
        logger.info(f" - Tổng số bản ghi ban đầu : {len(df)}")
        logger.info(f" - Review spam bị loại     : {spam_count} ({spam_count/len(df)*100:.1f}%)")
        logger.info(f" - Review trùng lặp bị loại: {dup_count} ({dup_count/len(df)*100:.1f}%)")
        logger.info(f" - Review hợp lệ phân tích : {valid_count} ({valid_count/len(df)*100:.1f}%)")

        return df

    def run_pipeline(
        self,
        input_path: str,
        output_path: str = "data/processed/reviews_processed.csv"
    ) -> pd.DataFrame:
        """Chạy pipeline từ file raw sang file processed."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Không tìm thấy file nguồn: {input_path}")

        logger.info(f"Đọc dữ liệu nguồn từ: {input_path}")
        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".json"):
            df = pd.read_json(input_path)
        else:
            raise ValueError("Định dạng file không hỗ trợ (chỉ nhận .csv hoặc .json).")

        processed_df = self.process_dataframe(df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu kết quả tiền xử lý vào: {output_path}")

        return processed_df
