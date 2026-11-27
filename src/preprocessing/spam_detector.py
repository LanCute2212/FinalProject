import os
import re
import json
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger("spam_detector")

class SpamDetector:
    """
    Module phát hiện review rác (spam) và review trùng lặp.
    Tuân thủ các Business Rules:
    - BR09: Review cần có nội dung hoặc thông tin đánh giá hợp lệ.
    - BR12: Review trùng lặp không được tính nhiều lần vào kết quả thống kê.
    - NFR16: Hạn chế ảnh hưởng của dữ liệu trùng lặp, thiếu hoặc không hợp lệ.
    """

    def __init__(self, spam_patterns_path: str = "data/dictionaries/spam_patterns.json"):
        data = self._load_json(spam_patterns_path)
        self.exact_default_phrases: List[str] = [p.lower() for p in data.get("exact_default_phrases", [])]
        self.spam_regex_patterns: List[str] = [p.lower() for p in data.get("spam_regex_patterns", [])]

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def check_spam(self, raw_text: str, clean_text: str) -> Tuple[bool, str]:
        """
        Kiểm tra 1 review có phải là spam/rác không.
        Trả về: (is_spam: bool, reason: str)
        """
        raw_str = "" if raw_text is None or pd.isna(raw_text) else str(raw_text).strip()
        clean_str = "" if clean_text is None or pd.isna(clean_text) else str(clean_text).strip().lower()
        missing_values = {"", "nan", "none", "null", "n/a", "na"}

        # 1. Rỗng hoặc chỉ chứa khoảng trắng
        if raw_str.lower() in missing_values or clean_str in missing_values:
            return True, "EMPTY_TEXT"

        # 2. Trùng 100% với các câu gợi ý mặc định của sàn TMĐT
        if clean_str in self.exact_default_phrases:
            return True, "DEFAULT_PLATFORM_TEMPLATE"

        # 3. Chứa các mẫu chữ săn xu, hình ảnh không liên quan
        for pattern in self.spam_regex_patterns:
            if pattern in clean_str or pattern in raw_str.lower():
                return True, "COIN_FARMING_SPAM"

        # Kiểm tra trên cả raw để TextCleaner chưa rút gọn mất tín hiệu lặp.
        # Ví dụ "aaaaaaaaaa" nếu kiểm tra sau khi clean chỉ còn "a".
        noise_candidates = (raw_str.lower(), clean_str)
        if any(
            len(compact := re.sub(r"\s+", "", candidate)) >= 10
            and len(set(compact)) <= 2
            for candidate in noise_candidates
        ):
            return True, "REPETITIVE_CHAR_NOISE"

        return False, "VALID"

    @staticmethod
    def check_low_information(clean_text: str) -> Tuple[bool, str]:
        """
        Phân biệt bình luận ít thông tin khỏi spam.
        Một từ duy nhất có nghĩa vẫn được xem là review hợp lệ.
        """
        clean_str = "" if clean_text is None or pd.isna(clean_text) else str(clean_text).strip().lower()
        missing_values = {"", "nan", "none", "null", "n/a", "na"}
        if clean_str in missing_values:
            return True, "EMPTY_TEXT"

        # Emoji tags carry sentiment information even though they use underscores.
        emoji_tags = {"_pos_emoji_", "_neg_emoji_", "_neu_emoji_"}
        words_without_tags = clean_str
        for tag in emoji_tags:
            words_without_tags = words_without_tags.replace(tag, " ")
        words = re.findall(r"[^\W_]+", words_without_tags, flags=re.UNICODE)
        meaningful_words = words + [tag for tag in emoji_tags if tag in clean_str.split()]
        if len(meaningful_words) < 1:
            return True, "NO_MEANINGFUL_TEXT"
        if len(meaningful_words) == 1:
            return True, "ONE_WORD_REVIEW"
        return False, "SUFFICIENT_TEXT"

    @classmethod
    def mark_duplicates(
        cls,
        df: pd.DataFrame,
        text_column: str = "comment_text_clean",
        duplicate_key_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Đánh dấu các review trùng lặp (BR12) bằng hàm băm MD5.
        Giữ lại bản ghi đầu tiên, các bản ghi sau gắn is_duplicate = True.
        """
        if text_column not in df.columns:
            df["is_duplicate"] = False
            return df

        # Tính hash MD5 cho từng dòng text đã clean
        def get_hash(text):
            if text is None or pd.isna(text):
                return ""
            normalized = str(text).strip().lower()
            if normalized in {"", "nan", "none", "null", "n/a", "na"}:
                return ""
            return hashlib.md5(normalized.encode("utf-8")).hexdigest()

        hashes = [get_hash(text) for text in df[text_column].tolist()]
        duplicate_candidates = [bool(value) for value in hashes]

        if duplicate_key_columns:
            available_keys = [column for column in duplicate_key_columns if column in df.columns]
            if available_keys:
                # Keep key columns separate instead of joining with a delimiter:
                # values containing "||" could otherwise collide.
                duplicate_frame = df[available_keys].copy().reset_index(drop=True)
                duplicate_frame["__content_hash__"] = hashes
                duplicate_frame["__has_content__"] = duplicate_candidates
                is_dup = (
                    duplicate_frame["__has_content__"]
                    & duplicate_frame.duplicated(
                        subset=[*available_keys, "__content_hash__"], keep="first"
                    )
                ).to_numpy(dtype=bool)
            else:
                duplicate_frame = pd.DataFrame({"hash": hashes, "has_content": duplicate_candidates})
                is_dup = (
                    duplicate_frame["has_content"]
                    & duplicate_frame.duplicated(subset=["hash"], keep="first")
                ).to_numpy(dtype=bool)
        else:
            duplicate_frame = pd.DataFrame({"hash": hashes, "has_content": duplicate_candidates})
            is_dup = (
                duplicate_frame["has_content"]
                & duplicate_frame.duplicated(subset=["hash"], keep="first")
            ).to_numpy(dtype=bool)

        df["is_duplicate"] = is_dup
        
        dup_count = int(sum(is_dup))
        percentage = dup_count / len(df) * 100 if len(df) else 0.0
        logger.info(f"Phát hiện {dup_count} review trùng lặp nội dung ({percentage:.2f}%).")
        return df
