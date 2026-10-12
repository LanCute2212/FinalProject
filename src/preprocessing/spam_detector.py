import os
import re
import json
import hashlib
import pandas as pd
from typing import Dict, List, Tuple
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
        raw_str = (raw_text or "").strip()
        clean_str = (clean_text or "").strip().lower()

        # 1. Rỗng hoặc chỉ chứa khoảng trắng
        if not raw_str:
            return True, "EMPTY_TEXT"

        # 2. Quá ngắn (ít hơn 3 ký tự hoặc ít hơn 2 từ)
        words = clean_str.split()
        if len(clean_str) < 3 or len(words) < 2:
            return True, "TOO_SHORT"

        # 3. Trùng 100% với các câu gợi ý mặc định của sàn TMĐT
        if clean_str in self.exact_default_phrases:
            return True, "DEFAULT_PLATFORM_TEMPLATE"

        # 4. Chứa các mẫu chữ săn xu, hình ảnh không liên quan
        for pattern in self.spam_regex_patterns:
            if pattern in clean_str or pattern in raw_str.lower():
                return True, "COIN_FARMING_SPAM"

        # 5. Ký tự lặp đơn điệu (độ đa dạng từ vựng quá thấp, vd: 'aaaaa', '11111')
        unique_chars = set(clean_str.replace(" ", ""))
        if len(clean_str) >= 10 and len(unique_chars) <= 2:
            return True, "REPETITIVE_CHAR_NOISE"

        return False, "VALID"

    @classmethod
    def mark_duplicates(cls, df: pd.DataFrame, text_column: str = "comment_text_clean") -> pd.DataFrame:
        """
        Đánh dấu các review trùng lặp (BR12) bằng hàm băm MD5.
        Giữ lại bản ghi đầu tiên, các bản ghi sau gắn is_duplicate = True.
        """
        if text_column not in df.columns:
            df["is_duplicate"] = False
            return df

        # Tính hash MD5 cho từng dòng text đã clean
        def get_hash(text):
            if not text or pd.isna(text):
                return ""
            return hashlib.md5(str(text).strip().encode("utf-8")).hexdigest()

        hashes = df[text_column].apply(get_hash)
        
        # Chỉ xét trùng lặp với các text có hash (không rỗng)
        is_dup = hashes.duplicated(keep="first") & (hashes != "")
        df["is_duplicate"] = is_dup
        
        dup_count = is_dup.sum()
        logger.info(f"Phát hiện {dup_count} review trùng lặp nội dung ({dup_count/len(df)*100:.2f}%).")
        return df
