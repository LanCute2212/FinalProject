import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from src.modeling.sentiment_evaluation import (
    evaluate_models,
    evaluate_nested_logistic,
    load_audited_training_data,
    train_and_save_best_model,
)


class SentimentEvaluationTests(unittest.TestCase):
    def test_evaluation_reports_baseline_and_candidates(self):
        rows = []
        samples = {
            "negative": ["hàng lỗi không dùng được", "giao chậm và thất vọng", "sản phẩm hỏng"],
            "neutral": ["sản phẩm bình thường", "đúng mô tả", "chất lượng tạm ổn"],
            "positive": ["rất tốt và hài lòng", "giao nhanh tuyệt vời", "sản phẩm đẹp"] * 2,
        }
        for label, texts in samples.items():
            rows.extend({"text": text, "label": label} for text in texts)
        result = evaluate_models(pd.DataFrame(rows), n_splits=2, n_repeats=1)
        names = {item["name"] for item in result["models"]}
        self.assertIn("majority_baseline", names)
        self.assertIn("logistic_word", names)
        self.assertIn("logistic_word_char", names)
        self.assertIn(result["best_model"], names - {"majority_baseline"})
        self.assertFalse(result["independent_test_available"])
        nested = result["nested_selection"]
        self.assertEqual(nested["protocol"], "nested_stratified_cv")
        self.assertEqual(nested["outer_repeats"], 1)
        self.assertEqual(set(nested["per_class_recall"]), {"negative", "neutral", "positive"})
        self.assertIn("class_weight", nested["full_data_inner_best_params"])

    def test_nested_logistic_selection_reports_outer_metrics_separately(self):
        rows = []
        for label, texts in {
            "negative": ["hàng lỗi không dùng được", "giao chậm thất vọng", "sản phẩm hỏng", "rất tệ", "không hài lòng", "đóng gói kém"],
            "neutral": ["sản phẩm bình thường", "đúng mô tả", "tạm ổn", "màu sắc bình thường", "chất lượng vừa", "chưa rõ"],
            "positive": ["rất tốt và hài lòng", "giao nhanh tuyệt vời", "sản phẩm đẹp", "rất đáng mua", "dùng tốt", "ưng ý"] * 2,
        }.items():
            rows.extend({"text": text, "label": label} for text in texts)
        result = evaluate_nested_logistic(
            pd.DataFrame(rows), outer_splits=3, inner_splits=2, random_state=17
        )
        self.assertEqual(result["protocol"], "nested_stratified_cv")
        self.assertEqual(result["outer_splits"], 3)
        self.assertEqual(len(result["outer_folds"]), 3)
        self.assertIn("macro_f1_mean", result["outer_reporting_metrics"])
        self.assertIn("macro_f1_std", result["outer_reporting_metrics"])
        self.assertIn("mean", result["per_class_recall"]["neutral"])
        self.assertIn("std", result["per_class_recall"]["neutral"])
        self.assertEqual(result["final_model_name"], "logistic_word_nested_tuned")
        self.assertFalse(result["independent_test_available"])

    def test_project_labels_are_audited_before_training(self):
        root = Path(__file__).resolve().parents[1]
        data, issues, summary = load_audited_training_data(
            root / "data/processed/manual_labels.csv",
            processed_path=root / "data/processed/reviews_processed.csv",
            reference_path=root / "data/processed/manual_labels_test.csv",
        )
        self.assertGreaterEqual(len(data), 3)
        self.assertEqual(set(data["label"].unique()), {"negative", "neutral", "positive"})
        self.assertFalse(summary["independent_test_available"])
        self.assertIn("training_dataset_fingerprint", summary)
        self.assertGreaterEqual(len(issues), 0)

    def test_saved_nested_model_is_marked_candidate_only(self):
        rows = []
        for label, texts in {
            "negative": ["hàng lỗi", "rất tệ", "không hài lòng", "giao chậm"],
            "neutral": ["bình thường", "đúng mô tả", "tạm ổn", "chưa rõ"],
            "positive": ["rất tốt", "hài lòng", "đáng mua", "dùng tốt"] * 2,
        }.items():
            rows.extend({"text": text, "label": label} for text in texts)
        data = pd.DataFrame(rows)
        evaluation = {
            "protocol": "repeated_stratified_cv",
            "best_model": "logistic_word",
            "models": [],
            "nested_selection": {
                "protocol": "nested_stratified_cv",
                "outer_splits": 2,
                "outer_repeats": 1,
                "inner_splits": 2,
                "inner_selection_metric": "macro_f1",
                "outer_reporting_metrics": {"macro_f1_mean": 0.5},
                "per_class_recall": {},
                "full_data_inner_best_params": {
                    "C": 0.5,
                    "class_weight": {"negative": 2.0, "neutral": 2.0, "positive": 1.0},
                },
                "final_model_name": "logistic_word_nested_tuned",
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "candidate.joblib"
            metadata_path = Path(temp_dir) / "candidate.json"
            metadata = train_and_save_best_model(
                data,
                evaluation,
                model_path=model_path,
                metadata_path=metadata_path,
                training_dataset_fingerprint="test-fingerprint",
            )
            self.assertTrue(model_path.exists())
            self.assertFalse(metadata["deployment_ready"])
            self.assertFalse(metadata["independent_test_available"])
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["model_name"], metadata["model_name"])


if __name__ == "__main__":
    unittest.main()
