import unittest

import pandas as pd

from src.features.aspect_discovery import discover_aspect_candidates


class AspectDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.reviews = pd.DataFrame([
            {
                "review_id": "r1",
                "platform": "tiki",
                "comment_text_segmented": "cốt_truyện hấp_dẫn chất_lượng tốt",
                "comment_text_raw": "Cốt truyện hấp dẫn, chất lượng tốt.",
                "is_valid_for_analysis": True,
            },
            {
                "review_id": "r2",
                "platform": "tiki",
                "comment_text_segmented": "cốt_truyện khá hay",
                "comment_text_raw": "Cốt truyện khá hay.",
                "is_valid_for_analysis": True,
            },
            {
                "review_id": "r1",
                "platform": "tiki",
                "comment_text_segmented": "cốt_truyện cốt_truyện",
                "comment_text_raw": "Cốt truyện.",
                "is_valid_for_analysis": True,
            },
            {
                "review_id": "r3",
                "platform": "tiki",
                "comment_text_segmented": "rất tốt nha",
                "comment_text_raw": "Rất tốt nha.",
                "is_valid_for_analysis": False,
            },
        ])

    def test_discovers_unmatched_phrases_by_distinct_reviews(self):
        candidates = discover_aspect_candidates(
            self.reviews,
            {"quality": ["chất lượng"]},
            min_frequency=2,
        ).set_index("keyword")

        self.assertEqual(candidates.loc["cốt truyện", "review_count"], 2)
        self.assertNotIn("chất lượng", candidates.index)
        self.assertNotIn("chất", candidates.index)
        self.assertNotIn("lượng", candidates.index)
        self.assertNotIn("tốt", candidates.index)
        self.assertEqual(len(candidates.loc["cốt truyện", "examples"]), 2)

    def test_frequency_threshold_filters_rare_candidates(self):
        candidates = discover_aspect_candidates(
            self.reviews,
            {"quality": ["chất lượng"]},
            min_frequency=3,
        )

        self.assertTrue(candidates.empty)

    def test_discards_sentiment_placeholder_tokens(self):
        reviews = pd.DataFrame([{
            "review_id": "emoji-only",
            "platform": "tiki",
            "comment_text_segmented": "pos_emoji pos_emoji",
            "is_valid_for_analysis": True,
        }])

        candidates = discover_aspect_candidates(
            reviews,
            {},
            min_frequency=1,
        )

        self.assertTrue(candidates.empty)

    def test_missing_ids_and_text_do_not_create_fake_candidates(self):
        reviews = pd.DataFrame([
            {
                "review_id": None,
                "platform": "tiki",
                "product_id": "p1",
                "comment_text_segmented": "cốt_truyện",
                "comment_text_raw": None,
                "is_valid_for_analysis": True,
            },
            {
                "review_id": None,
                "platform": "tiki",
                "product_id": "p2",
                "comment_text_segmented": "cốt_truyện",
                "comment_text_raw": None,
                "is_valid_for_analysis": True,
            },
        ])

        candidates = discover_aspect_candidates(
            reviews,
            {},
            min_frequency=2,
        ).set_index("keyword")

        self.assertEqual(candidates.loc["cốt truyện", "review_count"], 2)
        self.assertNotIn("nan", candidates.index)
        self.assertEqual(candidates.loc["cốt truyện", "examples"][0]["text"], "cốt truyện")


if __name__ == "__main__":
    unittest.main()
