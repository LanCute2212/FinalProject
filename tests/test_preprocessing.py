import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_preprocess import load_all_raw_data
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.spam_detector import SpamDetector


class TextNormalizationTests(unittest.TestCase):
    def test_default_dictionaries_load_outside_project_working_directory(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                normalizer = TextNormalizer()
                self.assertEqual(normalizer.normalize("sp"), "sản phẩm")
            finally:
                os.chdir(original_cwd)

    def test_canonical_words_are_not_expanded_or_mistransformed(self):
        normalizer = TextNormalizer()
        normalized = normalizer.normalize("ưng ý, mình đi qua nhà Mac, giá rẻ")
        self.assertIn("ưng ý", normalized)
        self.assertIn("mình", normalized)
        self.assertIn("qua", normalized)
        self.assertIn("mac", normalized)
        self.assertIn("rẻ", normalized)
        self.assertNotIn("ưng ý ý", normalized)
        self.assertNotIn("quá nhà", normalized)

    def test_teencode_phrase_variant_normalizes_without_partial_match(self):
        normalizer = TextNormalizer()
        self.assertEqual(normalizer.normalize("xịn sò"), "chất lượng tốt")

    def test_empty_teencode_dictionary_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            empty_dictionary = Path(temporary_directory) / "empty.json"
            empty_dictionary.write_text("{}", encoding="utf-8")
            normalizer = TextNormalizer(
                teencode_dict_path=str(empty_dictionary),
                emoji_dict_path=str(empty_dictionary),
            )
            self.assertEqual(normalizer.normalize("hello"), "hello")


class PreprocessingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = PreprocessingPipeline()

    def test_dataframe_processing_is_positional_even_with_duplicate_indices(self):
        frame = pd.DataFrame(
            {
                "comment_text": ["Giao hàng nhanh và tốt", "Sản phẩm bị lỗi"],
                "rating": [5, 1],
                "platform": ["tiki", "shopee"],
                "product_id": ["p1", "p1"],
            },
            index=[7, 7],
        )

        processed = self.pipeline.process_dataframe(frame)

        self.assertEqual(processed.index.tolist(), [7, 7])
        self.assertEqual(processed["comment_text_raw"].tolist(), frame["comment_text"].tolist())
        self.assertIn("giao hàng", processed.loc[7, "comment_text_clean"].iloc[0])
        self.assertIn("lỗi", processed.loc[7, "comment_text_clean"].iloc[1])

    def test_missing_raw_text_stays_missing_and_is_flagged_as_empty(self):
        frame = pd.DataFrame({"comment_text": [None, "   ", "nan"]})
        processed = self.pipeline.process_dataframe(frame)

        self.assertTrue(pd.isna(processed.loc[0, "comment_text_raw"]))
        self.assertEqual(processed.loc[0, "comment_text_clean"], "")
        self.assertTrue(processed["is_spam"].all())
        self.assertFalse(processed["is_valid_for_analysis"].any())

    def test_missing_raw_column_values_fall_back_positionally_to_source_text(self):
        frame = pd.DataFrame(
            {
                "comment_text": ["review thứ nhất", "review thứ hai"],
                "comment_text_raw": [None, "raw thứ hai"],
            },
            index=[9, 9],
        )
        processed = self.pipeline.process_dataframe(frame)
        self.assertEqual(processed["comment_text_raw"].tolist(), ["review thứ nhất", "raw thứ hai"])

    def test_missing_rating_markers_get_missing_status(self):
        frame = pd.DataFrame({"comment_text": ["có nội dung"], "rating": ["NaN"]})
        processed = self.pipeline.process_dataframe(frame)
        self.assertFalse(processed.loc[0, "rating_is_valid"])
        self.assertEqual(processed.loc[0, "rating_validation_status"], "MISSING_RATING")
        self.assertFalse(processed.loc[0, "is_valid_for_analysis"])
        self.assertTrue(processed.loc[0, "is_valid_for_text_analysis"])

    def test_run_pipeline_accepts_output_path_without_parent_directory(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                pd.DataFrame({"comment_text": ["Sản phẩm dùng ổn"], "rating": [4]}).to_csv(
                    "input.csv", index=False
                )
                result = self.pipeline.run_pipeline("input.csv", "output.csv")
                self.assertEqual(len(result), 1)
                self.assertTrue(Path("output.csv").exists())
            finally:
                os.chdir(original_cwd)

    def test_bad_rating_is_excluded_from_rating_analysis_but_kept_for_text_analysis(self):
        frame = pd.DataFrame(
            {
                "comment_text": ["Sản phẩm dùng rất tốt", "Giao hàng nhanh"],
                "rating": [7, "not-a-rating"],
            }
        )
        processed = self.pipeline.process_dataframe(frame)

        self.assertFalse(processed["rating_is_valid"].any())
        self.assertEqual(processed["rating_validation_status"].tolist(), ["INVALID_RATING", "INVALID_RATING"])
        self.assertFalse(processed["is_valid_for_analysis"].any())
        self.assertTrue(processed["is_valid_for_text_analysis"].all())

    def test_repetitive_noise_is_detected_before_cleaner_erases_the_signal(self):
        result = self.pipeline.process_single_text("aaaaaaaaaaaa")
        self.assertEqual(result["comment_text_clean"], "a")
        self.assertTrue(result["is_spam"])
        self.assertEqual(result["spam_reason"], "REPETITIVE_CHAR_NOISE")

    def test_empty_dataframe_has_complete_flags(self):
        processed = self.pipeline.process_dataframe(pd.DataFrame(columns=["comment_text", "rating"]))
        self.assertIn("rating_is_valid", processed.columns)
        self.assertIn("is_valid_for_text_analysis", processed.columns)
        self.assertEqual(len(processed), 0)


class SpamAndDuplicateTests(unittest.TestCase):
    def test_punctuation_only_is_low_information(self):
        self.assertEqual(SpamDetector.check_low_information("! , ? ."), (True, "NO_MEANINGFUL_TEXT"))

    def test_emoji_only_is_retained_as_a_sentiment_signal(self):
        self.assertEqual(SpamDetector.check_low_information("_pos_emoji_"), (True, "ONE_WORD_REVIEW"))

    def test_duplicate_keys_are_column_safe_and_support_duplicate_indices(self):
        frame = pd.DataFrame(
            {
                "platform": ["x||y", "x", "x||y", "x"],
                "product_id": ["z", "y||z", "z", "y||z"],
                "comment_text_clean": ["same text", "same text", "same text", "same text"],
            },
            index=[2, 2, 2, 2],
        )

        result = SpamDetector.mark_duplicates(
            frame,
            duplicate_key_columns=["platform", "product_id"],
        )

        self.assertEqual(result["is_duplicate"].tolist(), [False, False, True, True])

    def test_duplicate_detection_does_not_mark_missing_text_as_duplicate(self):
        frame = pd.DataFrame({"comment_text_clean": [None, None, "review", "review"]})
        result = SpamDetector.mark_duplicates(frame)
        self.assertEqual(result["is_duplicate"].tolist(), [False, False, False, True])


class RawDataMergeTests(unittest.TestCase):
    def test_deduplicates_within_platform_only_and_preserves_missing_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_directory = Path(temporary_directory)
            pd.DataFrame(
                {
                    "platform": ["tiki", "tiki", "tiki", "tiki"],
                    "review_id": ["same", "same", None, None],
                    "comment_text": ["old version", "latest version", "id-less one", "id-less two"],
                }
            ).to_csv(raw_directory / "a_raw.csv", index=False)
            pd.DataFrame(
                {
                    "platform": ["shopee", "tiki"],
                    "review_id": ["same", "nan"],
                    "comment_text": ["other platform", "literal nan ID"],
                }
            ).to_csv(raw_directory / "b_raw.csv", index=False)

            merged = load_all_raw_data(str(raw_directory))

        self.assertEqual(len(merged), 5)
        self.assertIn("latest version", merged["comment_text"].tolist())
        self.assertNotIn("old version", merged["comment_text"].tolist())
        self.assertIn("other platform", merged["comment_text"].tolist())
        self.assertIn("id-less one", merged["comment_text"].tolist())
        self.assertIn("id-less two", merged["comment_text"].tolist())


if __name__ == "__main__":
    unittest.main()
