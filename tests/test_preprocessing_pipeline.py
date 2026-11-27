import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.run_preprocess import run_full_preprocessing
from src.preprocessing.pipeline import PreprocessingPipeline


class PreprocessingPipelineTests(unittest.TestCase):
    def test_process_dataframe_preserves_rows_with_nonconsecutive_indexes(self):
        frame = pd.DataFrame(
            {
                "review_id": ["r1", "r2"],
                "comment_text": ["Sản phẩm dùng tốt", "Giao hàng khá nhanh"],
            },
            index=[0, 2],
        )

        processed = PreprocessingPipeline().process_dataframe(frame)

        self.assertEqual(list(processed.index), [0, 2])
        self.assertTrue(processed["is_spam"].notna().all())
        self.assertTrue(processed["is_duplicate"].notna().all())
        self.assertTrue(processed["is_valid_for_analysis"].notna().all())
        self.assertEqual(len(processed), 2)

    def test_run_full_preprocessing_can_skip_console_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            raw_dir.mkdir()
            pd.DataFrame(
                [{
                    "platform": "tiki",
                    "product_id": "p1",
                    "review_id": "r1",
                    "comment_text": "Sản phẩm dùng tốt",
                }]
            ).to_csv(raw_dir / "tiki_raw.csv", index=False)
            output_path = Path(temp_dir) / "processed.csv"

            with patch("scripts.run_preprocess.display_statistics") as display_statistics:
                processed = run_full_preprocessing(
                    raw_dir=str(raw_dir),
                    output_path=str(output_path),
                    sync_sql_server=False,
                    show_statistics=False,
                )

        display_statistics.assert_not_called()
        self.assertEqual(len(processed), 1)


if __name__ == "__main__":
    unittest.main()
