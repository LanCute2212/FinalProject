"""Benchmark and retrain the sentiment classifier from audited manual labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.sentiment_evaluation import (
    evaluate_models,
    load_audited_training_data,
    train_and_save_best_model,
    write_evaluation_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment models using audited manual labels.")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/manual_labels.csv")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/processed/manual_labels_test.csv")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/processed/reviews_processed.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/evaluation")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/sentiment_model.joblib")
    parser.add_argument("--metadata-path", type=Path, default=ROOT / "models/sentiment_model_metadata.json")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nested-outer-splits", type=int, default=5)
    parser.add_argument("--nested-inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data, issues, audit_summary = load_audited_training_data(
        args.labels, processed_path=args.processed, reference_path=args.reference
    )
    evaluation = evaluate_models(
        data,
        n_splits=args.splits,
        n_repeats=args.repeats,
        nested_outer_splits=args.nested_outer_splits,
        nested_inner_splits=args.nested_inner_splits,
        random_state=args.seed,
    )
    metadata = train_and_save_best_model(
        data,
        evaluation,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        training_dataset_fingerprint=audit_summary["training_dataset_fingerprint"],
        audit_summary=audit_summary,
        random_state=args.seed,
    )
    write_evaluation_artifacts(evaluation, output_dir=args.output_dir, audit_summary=audit_summary, issues=issues)
    nested = evaluation["nested_selection"]
    print(json.dumps({
        "exploratory_best_model": evaluation["best_model"],
        "nested_selection": {
            "protocol": nested["protocol"],
            "outer_splits": nested["outer_splits"],
            "inner_splits": nested["inner_splits"],
            "outer_reporting_metrics": nested["outer_reporting_metrics"],
            "per_class_recall": nested["per_class_recall"],
            "full_data_inner_best_params": nested["full_data_inner_best_params"],
            "independent_test_available": nested["independent_test_available"],
        },
        **audit_summary,
        "model_metadata": metadata,
    }, ensure_ascii=False, indent=2))
    print(f"Evaluation artifacts written under: {args.output_dir}")


if __name__ == "__main__":
    main()
