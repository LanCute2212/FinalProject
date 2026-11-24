import re
from collections import defaultdict
from typing import Dict, List

import pandas as pd


STOPWORDS = {
    "ạ", "anh", "bạn", "bị", "bởi", "các", "cái", "cho", "chứ", "của",
    "đã", "đang", "đây", "đó", "được", "em", "gì", "hay", "khi", "không",
    "là", "lắm", "mình", "một", "mua", "này", "nên", "nha", "nhiều",
    "như", "nhưng", "những", "nó", "nữa", "phẩm", "quá", "rất", "rồi",
    "sản", "sau", "sẽ", "thì", "thấy", "thật", "thôi", "tôi", "trong",
    "trên", "từ", "và", "vào", "về", "với", "xin", "shop", "hàng",
    "bán", "bản", "bọc", "cả", "cảm", "chưa", "chúc", "có", "còn",
    "cuốn", "dung", "dùng", "đầu", "để", "hơn", "hộ", "ích", "làm",
    "lần", "lòng", "lại", "mà", "mới", "mọi", "ngày", "nào", "nội",
    "phải", "ra", "sao", "thất", "thích", "tiếp", "tiki", "tục", "nhé",
    "vì", "ơn", "in", "hình", "quyển", "cũng", "thêm", "thể", "đến",
    "biết", "hiểu", "đúng", "tuy", "nhiên", "trang", "góc", "cửa", "đọc",
    "người", "sách",
}

OPINION_WORDS = {
    "cẩn", "đáng", "đắt", "đẹp", "hài", "hấp", "hữu", "khá", "hơi",
    "kém", "nhanh", "rẻ", "dẫn", "thận", "tệ", "tốt", "tuyệt", "ưng",
    "vọng", "vời", "xấu", "ổn", "chắc", "chắn",
}

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
PHRASE_BOUNDARY_STOPWORDS = {
    "ạ", "bạn", "cho", "của", "được", "em", "khi", "là", "mình", "này",
    "nên", "nha", "nhưng", "như", "quá", "rất", "thì", "thấy", "thôi",
    "tôi", "và", "với", "bán", "cả", "có", "cuốn", "dùng", "hơn",
    "làm", "lần", "mà", "mọi", "ngày", "ra", "sao", "nhé", "cửa", "hàng",
    "mới", "nào", "tiếp", "tiki", "tục", "ủng", "hộ", "còn", "lại", "vì",
    "tuy", "nhiên",
}


def _display_value(value, fallback: str = "") -> str:
    if pd.isna(value):
        return fallback
    return str(value)


def normalize_phrase(value: str) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value or "").lower().replace("_", " ")).strip()


def discover_aspect_candidates(
    reviews: pd.DataFrame,
    aspect_keywords: Dict[str, List[str]],
    min_frequency: int = 2,
    max_ngram: int = 3,
    example_limit: int = 3,
) -> pd.DataFrame:
    if min_frequency < 1:
        raise ValueError("Tần suất tối thiểu phải lớn hơn 0.")
    if max_ngram < 1:
        raise ValueError("Số từ tối đa trong candidate phải lớn hơn 0.")

    text_column = next(
        (column for column in ("comment_text_segmented", "comment_text_clean", "comment_text_raw") if column in reviews),
        None,
    )
    if text_column is None:
        raise ValueError("Dữ liệu cần có cột comment_text_segmented, comment_text_clean hoặc comment_text_raw.")

    frame = reviews.copy()
    if "is_valid_for_analysis" in frame:
        frame = frame[frame["is_valid_for_analysis"].fillna(False).astype(bool)]

    known_subphrases = set()
    for keywords in aspect_keywords.values():
        for keyword in keywords:
            keyword_tokens = TOKEN_PATTERN.findall(normalize_phrase(keyword))
            for size in range(1, len(keyword_tokens) + 1):
                for start in range(len(keyword_tokens) - size + 1):
                    known_subphrases.add(" ".join(keyword_tokens[start:start + size]))
    occurrences = defaultdict(dict)

    for row_index, row in frame.iterrows():
        text = normalize_phrase(row.get(text_column, ""))
        text = re.sub(r"\b(?:pos|neg|neu)\s+emoji\b", " ", text)
        tokens = TOKEN_PATTERN.findall(text)
        if not tokens:
            continue

        review_id = _display_value(row.get("review_id"), str(row_index)).strip()
        if not review_id or review_id.lower() in {"nan", "none"}:
            review_id = str(row_index)
        platform = _display_value(row.get("platform")).strip()
        product_id = _display_value(row.get("product_id")).strip()
        product_name = _display_value(row.get("product_name")).strip()
        product_key = product_id or product_name
        review_key = f"{platform}::{product_key}::{review_id}"
        raw_text = _display_value(row.get("comment_text_raw")).strip()
        clean_text = _display_value(row.get("comment_text_clean")).strip()
        record = {
            "review_id": review_id,
            "platform": platform,
            "product_name": product_name,
            "rating": row.get("rating", ""),
            "text": raw_text or clean_text or text,
        }

        phrases = set()
        for size in range(1, min(max_ngram, len(tokens)) + 1):
            for start in range(len(tokens) - size + 1):
                words = tokens[start:start + size]
                if any(word.isdigit() or len(word) < 2 for word in words):
                    continue
                if size == 1 and words[0] in STOPWORDS:
                    continue
                if size > 1 and (
                    words[0] in PHRASE_BOUNDARY_STOPWORDS
                    or words[-1] in PHRASE_BOUNDARY_STOPWORDS
                    or all(word in STOPWORDS for word in words)
                ):
                    continue
                if any(word in OPINION_WORDS for word in words):
                    continue
                phrase = " ".join(words)
                if phrase not in known_subphrases:
                    phrases.add(phrase)

        for phrase in phrases:
            occurrences[phrase][review_key] = record

    candidates = []
    for phrase, examples_by_review in occurrences.items():
        if len(examples_by_review) < min_frequency:
            continue
        examples = list(examples_by_review.values())[:example_limit]
        candidates.append({
            "keyword": phrase,
            "review_count": len(examples_by_review),
            "examples": examples,
        })

    candidates.sort(
        key=lambda item: (-item["review_count"], -len(item["keyword"].split()), item["keyword"])
    )
    return pd.DataFrame(candidates, columns=["keyword", "review_count", "examples"])
