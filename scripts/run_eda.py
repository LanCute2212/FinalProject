import argparse
import json
import os
import re
from typing import Dict, Iterable, List

import pandas as pd


DEFAULT_ASPECTS = "data/dictionaries/aspect_keywords.json"

POSITIVE_TERMS = ("tốt", "hài lòng", "ưng ý", "tuyệt vời", "đáng tiền", "mượt", "nhanh", "đẹp", "ổn")
NEGATIVE_TERMS = ("lỗi", "hỏng", "chậm", "tệ", "thất vọng", "trầy", "móp", "không tốt", "không ổn", "không bao giờ", "chập chờn")


def load_aspects(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_matches(text: str, keywords: Iterable[str]) -> List[str]:
    normalized = re.sub(r"[\s_]+", " ", str(text or "").lower()).strip()
    return [
        keyword for keyword in keywords
        if re.sub(r"[\s_]+", " ", keyword.lower()).strip() in normalized
    ]


def classify_contradiction(rating: int, text: str) -> str:
    normalized = str(text or "").lower()
    has_positive = any(term in normalized for term in POSITIVE_TERMS)
    has_negative = any(term in normalized for term in NEGATIVE_TERMS)
    if rating >= 4 and has_negative:
        return "HIGH_RATING_WITH_NEGATIVE_SIGNAL"
    if rating <= 2 and has_positive:
        return "LOW_RATING_WITH_POSITIVE_SIGNAL"
    return "NONE"


def build_report(df: pd.DataFrame, aspects: Dict[str, List[str]]) -> tuple[pd.DataFrame, Dict]:
    text_column = "comment_text_clean" if "comment_text_clean" in df else "comment_text_raw"
    if text_column not in df:
        raise ValueError("File đầu vào cần có cột comment_text_clean hoặc comment_text_raw.")
    if "rating" not in df:
        raise ValueError("File đầu vào cần có cột rating.")

    result = df.copy()
    result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    result["text_length_words"] = result[text_column].fillna("").astype(str).str.split().str.len()
    result["rating_text_contradiction"] = [
        classify_contradiction(int(rating), text) if pd.notna(rating) else "INVALID_RATING"
        for rating, text in zip(result["rating"], result[text_column])
    ]

    valid_mask = result.get("is_valid_for_analysis", pd.Series(True, index=result.index)).fillna(False).astype(bool)
    valid = result[valid_mask]
    aspect_counts = {}
    for aspect, keywords in aspects.items():
        match_column = f"aspect_{aspect}_matches"
        result[match_column] = result[text_column].apply(lambda text: ", ".join(find_matches(text, keywords)))
        aspect_counts[aspect] = int(result.loc[valid_mask, match_column].ne("").sum())

    contradiction_mask = result["rating_text_contradiction"].ne("NONE")
    summary = {
        "total_reviews": int(len(result)),
        "valid_reviews": int(len(valid)),
        "rating_distribution": {str(int(key)): int(value) for key, value in result["rating"].value_counts().sort_index().items() if pd.notna(key)},
        "word_count": {
            "mean": round(float(valid["text_length_words"].mean()), 2) if len(valid) else 0,
            "median": round(float(valid["text_length_words"].median()), 2) if len(valid) else 0,
            "min": int(valid["text_length_words"].min()) if len(valid) else 0,
            "max": int(valid["text_length_words"].max()) if len(valid) else 0,
        },
        "rating_text_contradictions": int(contradiction_mask.sum()),
        "contradiction_breakdown": {str(key): int(value) for key, value in result.loc[contradiction_mask, "rating_text_contradiction"].value_counts().items()},
        "aspect_coverage": {aspect: {"reviews": count, "rate_percent": round(count / len(valid) * 100, 2) if len(valid) else 0} for aspect, count in aspect_counts.items()},
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA cho dữ liệu review đã tiền xử lý")
    parser.add_argument("--input", default="data/processed/reviews_processed.csv")
    parser.add_argument("--output-dir", default="data/processed/eda")
    parser.add_argument("--aspects", default=DEFAULT_ASPECTS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = pd.read_csv(args.input)
    result, summary = build_report(data, load_aspects(args.aspects))
    detail_path = os.path.join(args.output_dir, "eda_review_details.csv")
    summary_path = os.path.join(args.output_dir, "eda_summary.json")
    result.to_csv(detail_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("=" * 64)
    print("EDA REPORT - CUSTOMER VOICE")
    print("=" * 64)
    print(f"Tổng review                 : {summary['total_reviews']}")
    print(f"Review hợp lệ               : {summary['valid_reviews']}")
    print(f"Nghịch lý rating - text     : {summary['rating_text_contradictions']}")
    print(f"Phân bố rating              : {summary['rating_distribution']}")
    print(f"Độ dài từ (mean / median)   : {summary['word_count']['mean']} / {summary['word_count']['median']}")
    print(f"Độ bao phủ aspect           : {summary['aspect_coverage']}")
    print(f"Chi tiết                    : {detail_path}")
    print(f"Tổng hợp                    : {summary_path}")


if __name__ == "__main__":
    main()