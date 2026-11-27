import unittest

from app import build_aspect_sentiment_rows


ASPECTS = {
    "quality": ["chất lượng"],
    "price": ["giá", "giá cả"],
    "delivery": ["giao hàng"],
}


class AspectSentimentHeuristicTests(unittest.TestCase):
    def test_contrast_assigns_sentiment_to_the_local_aspect_clause(self):
        rows = build_aspect_sentiment_rows(
            "Chất lượng tốt nhưng giao hàng chậm",
            ASPECTS,
        )

        self.assertEqual(
            {row["aspect"]: row["sentiment"] for row in rows},
            {"quality": "Tích cực", "delivery": "Tiêu cực"},
        )

    def test_basic_negation_flips_positive_word(self):
        rows = build_aspect_sentiment_rows("Chất lượng không tốt", ASPECTS)

        self.assertEqual(rows[0]["sentiment"], "Tiêu cực")

    def test_price_keyword_does_not_match_the_verb_phrase_danh_gia(self):
        rows = build_aspect_sentiment_rows("Tôi đánh giá sản phẩm rất tốt", ASPECTS)

        self.assertNotIn("price", {row["aspect"] for row in rows})

    def test_comma_separated_aspects_keep_their_own_polarity(self):
        rows = build_aspect_sentiment_rows("Chất lượng tốt, giá đắt", ASPECTS)

        self.assertEqual(
            {row["aspect"]: row["sentiment"] for row in rows},
            {"quality": "Tích cực", "price": "Tiêu cực"},
        )

    def test_same_aspect_with_both_polarities_is_mixed(self):
        rows = build_aspect_sentiment_rows(
            "Chất lượng tốt nhưng chất lượng chập chờn",
            ASPECTS,
        )

        self.assertEqual(rows[0]["sentiment"], "Hỗn hợp")

    def test_schema_and_metadata_are_preserved(self):
        rows = build_aspect_sentiment_rows(
            "Chất lượng tốt",
            ASPECTS,
            meta={"review_id": "r-1"},
        )

        self.assertEqual(
            set(rows[0]),
            {"aspect", "sentiment", "matched_keywords", "review_id"},
        )
        self.assertEqual(rows[0]["review_id"], "r-1")


if __name__ == "__main__":
    unittest.main()
