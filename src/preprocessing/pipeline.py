import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.spam_detector import SpamDetector
from src.preprocessing.tokenizer import VietnameseTokenizer
from src.utils.logger import get_logger

logger = get_logger("preprocessing_pipeline")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path: str) -> str:
    """Resolve default project resources even when a command runs elsewhere."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    rooted_candidate = PROJECT_ROOT / candidate
    return str(rooted_candidate if rooted_candidate.exists() else candidate)

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
            teencode_dict_path=_resolve_project_path(teencode_dict_path),
            emoji_dict_path=_resolve_project_path(emoji_dict_path)
        )
        self.spam_detector = SpamDetector(spam_patterns_path=_resolve_project_path(spam_patterns_path))
        self.tokenizer = VietnameseTokenizer()

    def process_single_text(self, text: str) -> Dict[str, Any]:
        """Xử lý tiền xử lý cho 1 chuỗi review đơn lẻ."""
        raw_text = "" if text is None or pd.isna(text) else str(text).strip()
        if raw_text.lower() in {"nan", "none", "null", "n/a", "na"}:
            raw_text = ""
        
        # Bước 1: Làm sạch cơ bản (HTML, URLs, lặp ký tự, NFC)
        cleaned_text = self.cleaner.clean(raw_text)
        
        # Bước 2: Chuẩn hóa teencode và emoji
        normalized_text = self.normalizer.normalize(cleaned_text, handle_emoji="tag", lower=True)
        
        # Bước 3: Kiểm tra spam/rác
        is_spam, spam_reason = self.spam_detector.check_spam(raw_text, normalized_text)
        low_information, low_information_reason = self.spam_detector.check_low_information(normalized_text)
        
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
            "spam_reason": spam_reason,
            "low_information": low_information,
            "low_information_reason": low_information_reason
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "comment_text") -> pd.DataFrame:
        """
        Xử lý toàn bộ DataFrame chứa review.
        """
        df = df.copy()
        logger.info(f"Bắt đầu pipeline tiền xử lý cho {len(df)} bản ghi...")

        if df.empty:
            for column, default in {
                "comment_text_raw": "",
                "comment_text_clean": "",
                "comment_text_segmented": "",
                "word_count": 0,
                "char_count": 0,
                "is_spam": False,
                "spam_reason": "EMPTY_DATA",
                "low_information": True,
                "low_information_reason": "EMPTY_DATA",
                "rating_is_valid": False,
                "rating_validation_status": "MISSING_RATING",
                "is_duplicate": False,
                "is_valid_for_text_analysis": False,
                "is_valid_for_analysis": False,
                "filter_status": "EMPTY_DATA",
            }.items():
                df[column] = pd.Series(dtype=type(default))
            return df

        # Đảm bảo có cột text gốc
        if text_column not in df.columns:
            raise ValueError(f"Không tìm thấy cột '{text_column}' trong DataFrame.")

        # Bảo toàn giá trị nguồn, gồm cả null; không đổi null thành chuỗi "nan".
        if "comment_text_raw" not in df.columns:
            df["comment_text_raw"] = df[text_column].copy()
        else:
            raw_values = df["comment_text_raw"].tolist()
            source_values = df[text_column].tolist()
            raw_values = [
                source if value is None or pd.isna(value) or str(value).strip() == "" else value
                for value, source in zip(raw_values, source_values)
            ]
            df["comment_text_raw"] = raw_values

        # Áp dụng từng bước theo thứ tự dòng, không căn chỉnh theo index có thể trùng.
        results = [self.process_single_text(t) for t in df["comment_text_raw"].tolist()]

        # Ghép kết quả vào DataFrame chính
        for col in results[0]:
            df[col] = [result[col] for result in results]

        # Chất lượng rating độc lập với khả năng phân tích nội dung văn bản.
        if "rating" in df.columns:
            numeric_rating = pd.to_numeric(df["rating"], errors="coerce")
            rating_is_valid = numeric_rating.between(1, 5) & numeric_rating.mod(1).eq(0)
            df["rating_is_valid"] = rating_is_valid.fillna(False).to_numpy(dtype=bool)
            df["rating_validation_status"] = [
                "VALID_RATING" if valid else (
                    "MISSING_RATING" if pd.isna(value) or str(value).strip().lower()
                    in {"", "nan", "none", "null", "n/a", "na"}
                    else "INVALID_RATING"
                )
                for value, valid in zip(df["rating"].tolist(), df["rating_is_valid"].tolist())
            ]
        else:
            df["rating_is_valid"] = False
            df["rating_validation_status"] = "MISSING_RATING"

        # Đánh dấu trùng lặp (BR12)
        df = self.spam_detector.mark_duplicates(
            df,
            text_column="comment_text_clean",
            duplicate_key_columns=["platform", "product_id"],
        )

        # Text sentiment có thể chạy độc lập với rating; các báo cáo rating và
        # feature labels chỉ dùng bản ghi có rating hợp lệ.
        df["is_valid_for_text_analysis"] = (
            ~df["is_spam"].to_numpy(dtype=bool)
            & ~df["is_duplicate"].to_numpy(dtype=bool)
        )
        df["is_valid_for_analysis"] = (
            df["is_valid_for_text_analysis"].to_numpy(dtype=bool)
            & df["rating_is_valid"].to_numpy(dtype=bool)
        )

        # Gán lý do loại bỏ rõ ràng
        df["filter_status"] = [
            f"SPAM: {reason}" if spam else "DUPLICATE_REVIEW" if duplicate else (
                "INVALID_RATING" if not rating_valid else "VALID"
            )
            for spam, reason, duplicate, rating_valid in zip(
                df["is_spam"].tolist(), df["spam_reason"].tolist(),
                df["is_duplicate"].tolist(), df["rating_is_valid"].tolist()
            )
        ]

        valid_count = df["is_valid_for_analysis"].sum()
        spam_count = df["is_spam"].sum()
        dup_count = df["is_duplicate"].sum()

        logger.info(f"=== BÁO CÁO TIỀN XỬ LÝ DỮ LIỆU ===")
        logger.info(f" - Tổng số bản ghi ban đầu : {len(df)}")
        denominator = len(df) or 1
        logger.info(f" - Review spam bị loại     : {spam_count} ({spam_count/denominator*100:.1f}%)")
        logger.info(f" - Review trùng lặp bị loại: {dup_count} ({dup_count/denominator*100:.1f}%)")
        logger.info(f" - Review hợp lệ phân tích : {valid_count} ({valid_count/denominator*100:.1f}%)")

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
        if Path(input_path).suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        elif Path(input_path).suffix.lower() == ".json":
            df = pd.read_json(input_path)
        else:
            raise ValueError("Định dạng file không hỗ trợ (chỉ nhận .csv hoặc .json).")

        processed_df = self.process_dataframe(df)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu kết quả tiền xử lý vào: {output_path}")

        return processed_df
