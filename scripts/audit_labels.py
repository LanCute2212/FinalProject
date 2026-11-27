"""Create a read-only quality audit of sentiment label files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_quality import audit_label_datasets


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"review_id": "string"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit labels and overlap without editing source CSV files."
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/manual_labels.csv")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/processed/manual_labels_test.csv")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/reviews_processed.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/quality")
    args = parser.parse_args()

    train = _read_csv(args.train)
    reference = _read_csv(args.reference) if args.reference.exists() else None
    processed = _read_csv(args.processed) if args.processed.exists() else None
    report = audit_label_datasets(train, reference=reference, processed=processed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for key, filename in (
        ("issues", "issues.csv"),
        ("training_audit", "training_labels.csv"),
        ("training_labels", "model_training_candidates.csv"),
        ("review_queue", "review_queue.csv"),
        ("annotation_candidates", "annotation_candidates.csv"),
    ):
        report[key].to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Audit files written under: {args.output_dir}")
    print("Source label files were read only; legacy reference data is not an independent test set.")


if __name__ == "__main__":
    main()
