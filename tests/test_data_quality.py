import unittest

import pandas as pd

from src.data_quality import (
    audit_label_datasets,
    dataset_fingerprint,
    prepare_training_labels,
    text_fingerprint,
)


def train_rows():
    return pd.DataFrame([
        {"review_id": "r1", "text": "Tốt\n lắm", "text_model": "tốt lắm", "sentiment_manual": "Tích cực", "sentiment_source": "manual"},
        {"review_id": "r2", "text": "Review bị trùng", "text_model": "review bị trùng", "sentiment_manual": "Tiêu cực", "sentiment_source": "manual"},
        {"review_id": "r3", "text": "review bị trùng", "text_model": "review bị trùng", "sentiment_manual": "Tích cực", "sentiment_source": "manual"},
        {"review_id": "r4", "text": "Nhãn AI", "text_model": "nhãn ai", "sentiment_manual": "Tiêu cực", "sentiment_source": "ai"},
    ])


class DataQualityTests(unittest.TestCase):
    def test_fingerprint_ignores_unicode_form_case_and_whitespace(self):
        self.assertEqual(text_fingerprint("Tốt  lắm"), text_fingerprint("tốt\nlắm"))
        self.assertEqual(text_fingerprint("ế"), text_fingerprint("e\u0302\u0301"))

    def test_training_filters_to_explicit_manual_and_quarantines_text_conflict(self):
        processed = pd.DataFrame({
            "review_id": ["r1", "r2", "r3", "r4"],
            "platform": ["tiki"] * 4,
            "comment_text_raw": ["Tốt lắm", "Review bị trùng", "review bị trùng", "Nhãn AI"],
        })
        usable, issues = prepare_training_labels(train_rows(), processed=processed)
        self.assertEqual(usable["review_id"].astype(str).tolist(), ["r1"])
        self.assertTrue((usable["canonical_platform"] == "tiki").all())
        self.assertEqual(set(issues["issue_code"]), {"conflicting_connected_labels_or_text"})

    def test_consistent_duplicate_rows_collapse_deterministically(self):
        train = pd.DataFrame([
            {"review_id": "r1", "text": "Rất ổn", "sentiment_manual": "Tích cực", "sentiment_source": "manual"},
            {"review_id": "r2", "text": "rất   ổn", "sentiment_manual": "Tích cực", "sentiment_source": "manual"},
        ])
        usable, issues = prepare_training_labels(train)
        self.assertEqual(len(usable), 1)
        self.assertEqual(usable.iloc[0]["review_id"], "r1")
        self.assertIn("duplicate_consistent", set(issues["issue_code"]))

    def test_legacy_reference_is_candidate_evidence_not_gold_or_test(self):
        train = pd.DataFrame([
            {"review_id": "r1", "text": "Rất ổn", "sentiment_manual": "Tích cực", "sentiment_source": "manual"},
        ])
        reference = pd.DataFrame([
            {"review_id": "r1", "platform": "tiki", "comment_text_raw": "Rất ổn", "sentiment_manual": "Tiêu cực", "sentiment_review": "Tích cực", "ai_review_status": "completed"},
            {"review_id": "r2", "platform": "tiki", "comment_text_raw": "Chưa có nhãn", "sentiment_manual": None, "sentiment_review": "Tiêu cực", "ai_review_status": "completed"},
        ])
        report = audit_label_datasets(train, reference=reference)
        self.assertEqual(len(report["training_labels"]), 0)
        self.assertEqual(len(report["annotation_candidates"]), 2)
        self.assertFalse(report["summary"]["legacy_reference_is_independent_test"])
        self.assertTrue(report["annotation_candidates"]["human_confirmation_required"].all())
        self.assertNotIn("sentiment_review", report["training_labels"].columns)
        self.assertIn("reference_label_conflict", set(report["issues"]["issue_code"]))

    def test_ambiguous_platform_id_is_quarantined(self):
        train = pd.DataFrame([
            {"review_id": "same", "text": "Không khớp", "sentiment_manual": "Tích cực", "sentiment_source": "manual"},
        ])
        processed = pd.DataFrame({
            "review_id": ["same", "same"],
            "platform": ["tiki", "shopee"],
            "comment_text_raw": ["nội dung A", "nội dung B"],
        })
        usable, issues = prepare_training_labels(train, processed=processed)
        self.assertTrue(usable.empty)
        self.assertIn("ambiguous_platform_match", set(issues["issue_code"]))

    def test_dataset_fingerprint_independent_of_row_order_and_index(self):
        first = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=[9, 4])
        second = first.iloc[::-1].reset_index(drop=True)
        self.assertEqual(dataset_fingerprint(first), dataset_fingerprint(second))
        second.loc[0, "b"] = "changed"
        self.assertNotEqual(dataset_fingerprint(first), dataset_fingerprint(second))


if __name__ == "__main__":
    unittest.main()
