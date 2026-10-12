import os
import re
import json
from typing import Dict, Optional, List
from src.utils.logger import get_logger

logger = get_logger("text_normalizer")

class TextNormalizer:
    """
    Module chuẩn hóa văn bản tiếng Việt cho E-commerce:
    - Thay thế từ viết tắt, teencode tiếng Việt thành từ chuẩn
    - Chuẩn hóa hoặc ánh xạ Emoji sang token cảm xúc
    - Giữ lại dấu câu logic quan trọng cho Aspect Analysis (. , ! ? nhưng)
    """

    def __init__(
        self,
        teencode_dict_path: str = "data/dictionaries/teencode_dict.json",
        emoji_dict_path: str = "data/dictionaries/emoji_dict.json"
    ):
        self.teencode_dict: Dict[str, str] = self._load_json(teencode_dict_path)
        self.emoji_dict: Dict[str, List[str]] = self._load_json(emoji_dict_path)
        
        # Tạo regex biên từ cho teencode để thay thế chính xác từng từ đơn
        # Sắp xếp các cụm từ dài trước để tránh match từ ngắn trước (vd: "không phải" trước "không")
        sorted_keys = sorted(self.teencode_dict.keys(), key=lambda x: len(x), reverse=True)
        escaped_keys = [re.escape(k) for k in sorted_keys]
        self.teencode_regex = re.compile(r"(?<!\w)(" + "|".join(escaped_keys) + r")(?!\w)", re.IGNORECASE)

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"Không tìm thấy file từ điển tại {path}. Dùng từ điển rỗng.")
        return {}

    def normalize_teencode(self, text: str) -> str:
        """Thay thế teencode và từ viết tắt bằng từ tiếng Việt chuẩn."""
        if not text or not self.teencode_dict:
            return text

        def replace_match(match):
            word = match.group(0).lower()
            return self.teencode_dict.get(word, match.group(0))

        return self.teencode_regex.sub(replace_match, text)

    def normalize_emoji(self, text: str, mode: str = "tag") -> str:
        """
        Chuẩn hóa Emoji trong review:
        - mode='tag': Chuyển thành token đặc biệt (_pos_emoji_, _neg_emoji_) giúp mô hình nhận biết cảm xúc.
        - mode='remove': Loại bỏ toàn bộ emoji.
        - mode='keep': Giữ nguyên emoji nhưng tách khoảng trắng xung quanh.
        """
        if not text or not self.emoji_dict:
            return text

        if mode == "tag":
            pos_list = self.emoji_dict.get("positive", [])
            neg_list = self.emoji_dict.get("negative", [])
            neu_list = self.emoji_dict.get("neutral", [])

            for emo in pos_list:
                text = text.replace(emo, " _pos_emoji_ ")
            for emo in neg_list:
                text = text.replace(emo, " _neg_emoji_ ")
            for emo in neu_list:
                text = text.replace(emo, " _neu_emoji_ ")

        elif mode == "remove":
            for category in self.emoji_dict.values():
                for emo in category:
                    text = text.replace(emo, " ")

        # Chuẩn hóa lại khoảng trắng sau khi thay thế
        return re.sub(r"\s+", " ", text).strip()

    def normalize(self, text: str, handle_emoji: str = "tag", lower: bool = True) -> str:
        """Thực hiện toàn bộ chuỗi chuẩn hóa ngôn ngữ."""
        if not text:
            return ""

        if lower:
            text = text.lower()

        # 1. Chuẩn hóa emoji
        text = self.normalize_emoji(text, mode=handle_emoji)

        # 2. Chuẩn hóa teencode
        text = self.normalize_teencode(text)

        # 3. Chuẩn hóa khoảng trắng xung quanh dấu câu để giữ cấu trúc mệnh đề
        text = re.sub(r"\s*([,.;!?])\s*", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text
