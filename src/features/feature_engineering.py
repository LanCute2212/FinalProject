import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


SENTIMENT_LABELS = {
    "negative": "Tiêu cực",
    "neutral": "Trung tính",
    "positive": "Tích cực",
}


class FeatureEngineeringPipeline:
    """Tạo feature baseline cho sentiment và aspect từ review đã preprocess."""

    def __init__(self, aspect_keywords_path: str = "data/dictionaries/aspect_keywords.json"):
        self.aspect_keywords = self._load_aspect_keywords(aspect_keywords_path)
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=15000,
        )

    @staticmethod
    def _load_aspect_keywords(path: str) -> Dict[str, List[str]]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _text_column(df: pd.DataFrame) -> str:
        for column in ("comment_text_segmented", "comment_text_clean", "comment_text_raw"):
            if column in df.columns:
                return column
        raise ValueError("Dữ liệu cần có comment_text_segmented, comment_text_clean hoặc comment_text_raw.")

    @staticmethod
    def sentiment_from_rating(rating: float) -> str:
        if pd.isna(rating) or not 1 <= float(rating) <= 5:
            raise ValueError(f"Rating không hợp lệ: {rating}")
        if rating <= 2:
            return "negative"
        if rating == 3:
            return "neutral"
        return "positive"

    @staticmethod
    def _find_matches(text: str, keywords: List[str]) -> List[str]:
        normalized = re.sub(r"[\s_]+", " ", str(text or "").lower()).strip()
        return [
            keyword for keyword in keywords
            if re.sub(r"[\s_]+", " ", keyword.lower()).strip() in normalized
        ]

    def build_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "rating" not in df.columns:
            raise ValueError("Dữ liệu cần có cột rating để tạo nhãn sentiment.")
        labels = pd.DataFrame(index=df.index)
        labels["review_id"] = df.get("review_id", pd.Series(df.index, index=df.index)).astype(str)
        labels["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        labels["sentiment"] = labels["rating"].apply(self.sentiment_from_rating)
        labels["sentiment_name"] = labels["sentiment"].map(SENTIMENT_LABELS)

        text_column = self._text_column(df)
        text = df[text_column].fillna("").astype(str)
        for aspect, keywords in self.aspect_keywords.items():
            labels[f"aspect_{aspect}"] = text.apply(lambda value: int(bool(self._find_matches(value, keywords))))
            labels[f"aspect_{aspect}_matches"] = text.apply(lambda value: ", ".join(self._find_matches(value, keywords)))
        return labels.reset_index(drop=True), text.reset_index(drop=True)

    def build_features(self, texts: pd.Series):
        word_matrix = self.word_vectorizer.fit_transform(texts)
        char_matrix = self.char_vectorizer.fit_transform(texts)
        return word_matrix, char_matrix

    def run(self, input_path: str, output_dir: str = "data/processed/features") -> Dict[str, object]:
        data = pd.read_csv(input_path)
        if "is_valid_for_analysis" in data.columns:
            valid_mask = data["is_valid_for_analysis"].fillna(False).astype(bool)
            data = data.loc[valid_mask].reset_index(drop=True)
        if data.empty:
            raise ValueError("Không có review hợp lệ để tạo feature.")

        labels, texts = self.build_labels(data)
        word_matrix, char_matrix = self.build_features(texts)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        labels.to_csv(output / "labels.csv", index=False, encoding="utf-8-sig")
        save_npz(output / "tfidf_word_ngrams.npz", word_matrix)
        save_npz(output / "tfidf_char_ngrams.npz", char_matrix)
        joblib.dump(self.word_vectorizer, output / "tfidf_word_vectorizer.joblib")
        joblib.dump(self.char_vectorizer, output / "tfidf_char_vectorizer.joblib")
        pd.DataFrame({"review_id": labels["review_id"], "text": texts}).to_csv(
            output / "feature_texts.csv", index=False, encoding="utf-8-sig"
        )

        metadata = {
            "source": input_path,
            "valid_reviews": int(len(data)),
            "sentiment_mapping": {"1-2": "negative", "3": "neutral", "4-5": "positive"},
            "word_tfidf": {"shape": list(word_matrix.shape), "features": len(self.word_vectorizer.get_feature_names_out())},
            "char_tfidf": {"shape": list(char_matrix.shape), "features": len(self.char_vectorizer.get_feature_names_out())},
            "aspect_names": list(self.aspect_keywords),
            "sentiment_distribution": {str(key): int(value) for key, value in labels["sentiment"].value_counts().items()},
        }
        with open(output / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        return metadata