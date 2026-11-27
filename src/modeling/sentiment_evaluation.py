"""Reproducible sentiment model benchmarking for the Model Lab.

The evaluator consumes only rows accepted by :func:`prepare_training_labels`.
It reports repeated stratified CV for exploratory model comparisons and nested
stratified CV for Logistic Regression hyperparameter selection. The project has
no clean, independent labelled test set, so metrics are development estimates,
not a claim about production performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.data_quality import dataset_fingerprint, prepare_training_labels


LABELS = ("negative", "neutral", "positive")
LABEL_MAP = {"Tiêu cực": "negative", "Trung tính": "neutral", "Tích cực": "positive"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    algorithm: str
    features: str


MODEL_SPECS = (
    ModelSpec("logistic_word", "logistic_regression", "word"),
    ModelSpec("logistic_word_char", "logistic_regression", "word_char"),
    ModelSpec("linear_svm_word", "linear_svc", "word"),
    ModelSpec("multinomial_nb_word", "multinomial_nb", "word"),
)

# Prespecified candidates: the balanced option and a small grid that increases
# the minority-class penalty while lowering the dominant positive-class weight.
# This grid is selected only inside each outer training fold.
LOGISTIC_TUNING_GRID = {
    "C": (0.5, 1.0, 2.0),
    "class_weight": (
        "balanced",
        {"negative": 8.0, "neutral": 8.0, "positive": 0.5},
        {"negative": 12.0, "neutral": 12.0, "positive": 0.5},
        {"negative": 8.0, "neutral": 12.0, "positive": 0.5},
        {"negative": 12.0, "neutral": 8.0, "positive": 0.5},
    ),
}


def _text_column(frame: pd.DataFrame) -> pd.Series:
    """Return stable model text, preferring the preprocessed representation."""
    if "text_model" in frame:
        text = frame["text_model"].fillna("").astype(str)
    elif "canonical_text" in frame:
        text = frame["canonical_text"].fillna("").astype(str)
    elif "text" in frame:
        text = frame["text"].fillna("").astype(str)
    else:
        raise ValueError("Training data must contain text, canonical_text, or text_model.")
    return text


def load_audited_training_data(
    labels_path: str | Path,
    *,
    processed_path: str | Path | None = None,
    reference_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load labels and return audited rows, issues, and an audit summary.

    Legacy reference labels are passed to the audit only for overlap/conflict
    detection. They are never appended to the training data.
    """
    labels_path = Path(labels_path)
    train = pd.read_csv(labels_path, encoding="utf-8-sig", dtype={"review_id": "string"})
    processed = None
    reference = None
    if processed_path is not None and Path(processed_path).exists():
        processed = pd.read_csv(processed_path, encoding="utf-8-sig", dtype={"review_id": "string"})
    if reference_path is not None and Path(reference_path).exists():
        reference = pd.read_csv(reference_path, encoding="utf-8-sig", dtype={"review_id": "string"})

    audited, issues = prepare_training_labels(train, reference=reference, processed=processed)
    if audited.empty:
        raise ValueError("Audit không còn mẫu nhãn thủ công hợp lệ để huấn luyện.")

    data = audited.copy()
    data["text"] = _text_column(data)
    data["label"] = data["sentiment_label"].map(lambda value: str(value).strip().casefold())
    data = data[data["label"].isin(LABELS) & data["text"].str.strip().ne("")].reset_index(drop=True)
    if data.empty:
        raise ValueError("Audit không còn mẫu có nội dung và nhãn cảm xúc hợp lệ.")

    counts = data["label"].value_counts().reindex(LABELS, fill_value=0)
    summary = {
        "source_labels": str(labels_path),
        "training_samples": int(len(data)),
        "class_distribution": {label: int(counts[label]) for label in LABELS},
        "audit_issue_rows": int(len(issues)),
        # Keep the fingerprint shape compatible with the Model Lab UI, which
        # identifies examples by review_id plus the exact model text/label.
        "training_dataset_fingerprint": dataset_fingerprint(
            data[["review_id", "text", "label"]]
            if "review_id" in data
            else data[["text", "label"]]
        ),
        "independent_test_available": False,
        "legacy_reference_is_independent_test": False,
    }
    return data, issues.reset_index(drop=True), summary


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=40000,
    )


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=30000,
    )


def make_model(spec: ModelSpec, *, random_state: int = 42) -> Pipeline:
    if spec.features == "word":
        features: Any = _word_vectorizer()
    elif spec.features == "word_char":
        features = FeatureUnion(
            [("word", _word_vectorizer()), ("char", _char_vectorizer())]
        )
    else:
        raise ValueError(f"Unknown feature set: {spec.features}")

    if spec.algorithm == "logistic_regression":
        classifier: Any = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    elif spec.algorithm == "linear_svc":
        classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=random_state)
    elif spec.algorithm == "multinomial_nb":
        classifier = MultinomialNB(alpha=0.5)
    else:
        raise ValueError(f"Unknown algorithm: {spec.algorithm}")
    return Pipeline([("features", features), ("classifier", classifier)])


def _fold_metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
        ),
    }


def _summarize(values: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in values[0]:
        array = np.asarray([item[key] for item in values], dtype=float)
        result[f"{key}_mean"] = float(array.mean())
        result[f"{key}_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return result


def _evaluate_constant_baseline(y: np.ndarray, splitter: RepeatedStratifiedKFold) -> dict[str, Any]:
    folds: list[dict[str, float]] = []
    aggregate = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    majority = str(pd.Series(y).value_counts().idxmax())
    for _, test_index in splitter.split(np.zeros(len(y)), y):
        predicted = np.full(len(test_index), majority, dtype=object)
        true = y[test_index]
        folds.append(_fold_metric(true, predicted))
        aggregate += confusion_matrix(true, predicted, labels=list(LABELS))
    return {
        "name": "majority_baseline",
        "algorithm": "majority_class",
        "features": "none",
        "majority_label": majority,
        "folds": len(folds),
        "metrics": _summarize(folds),
        "confusion_matrix": aggregate.tolist(),
        "per_class": {},
    }


def _evaluate_spec(
    spec: ModelSpec,
    texts: pd.Series,
    labels: np.ndarray,
    splitter: RepeatedStratifiedKFold,
    *,
    random_state: int,
) -> dict[str, Any]:
    folds: list[dict[str, float]] = []
    aggregate = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    class_values: dict[str, list[dict[str, float]]] = {label: [] for label in LABELS}
    x_values = texts.to_numpy(dtype=object)
    for train_index, test_index in splitter.split(x_values, labels):
        model = make_model(spec, random_state=random_state)
        model.fit(x_values[train_index], labels[train_index])
        predicted = model.predict(x_values[test_index])
        true = labels[test_index]
        folds.append(_fold_metric(true, predicted))
        aggregate += confusion_matrix(true, predicted, labels=list(LABELS))
        precision, recall, f1, support = precision_recall_fscore_support(
            true, predicted, labels=list(LABELS), zero_division=0
        )
        for index, label in enumerate(LABELS):
            class_values[label].append(
                {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": float(support[index]),
                }
            )
    per_class: dict[str, dict[str, float]] = {}
    for label, rows in class_values.items():
        per_class[label] = {}
        for key in ("precision", "recall", "f1", "support"):
            values = np.asarray([row[key] for row in rows], dtype=float)
            per_class[label][f"{key}_mean"] = float(values.mean())
            per_class[label][f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "name": spec.name,
        "algorithm": spec.algorithm,
        "features": spec.features,
        "folds": len(folds),
        "metrics": _summarize(folds),
        "confusion_matrix": aggregate.tolist(),
        "per_class": per_class,
    }


def _make_logistic_word_model(*, C: float = 1.0, class_weight: Any = "balanced", random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("features", _word_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _class_weight_for_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): float(weight) for key, weight in value.items()}
    return value


def evaluate_nested_logistic(
    data: pd.DataFrame,
    *,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Select LR hyperparameters inside inner folds and report outer-fold results.

    The reported outer folds estimate the full model-selection procedure. A
    separate inner CV fit on all audited rows chooses hyperparameters for the
    final model artifact; that full-data inner score is not reported as test
    performance.
    """
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Nested evaluation data must contain text and label columns.")
    texts = data["text"].fillna("").astype(str).to_numpy(dtype=object)
    labels = data["label"].fillna("").astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    if any(label not in counts or int(counts[label]) < 3 for label in LABELS):
        raise ValueError("Mỗi lớp cần ít nhất 3 mẫu để chạy nested cross-validation.")

    outer_count = min(max(2, int(outer_splits)), int(counts.min()))
    outer_cv = StratifiedKFold(n_splits=outer_count, shuffle=True, random_state=random_state)
    scorer = make_scorer(
        f1_score, average="macro", labels=list(LABELS), zero_division=0
    )
    class_weights = list(LOGISTIC_TUNING_GRID["class_weight"])
    outer_rows: list[dict[str, Any]] = []
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    class_recalls: dict[str, list[float]] = {label: [] for label in LABELS}

    for fold_number, (train_index, test_index) in enumerate(outer_cv.split(texts, labels), start=1):
        y_train = labels[train_index]
        train_counts = pd.Series(y_train).value_counts()
        fold_inner_count = min(int(inner_splits), int(train_counts.min()))
        if fold_inner_count < 2:
            raise ValueError("Không đủ nhãn trong outer-train để chạy inner cross-validation.")
        inner_cv = StratifiedKFold(
            n_splits=fold_inner_count,
            shuffle=True,
            random_state=random_state + fold_number,
        )
        search = GridSearchCV(
            estimator=_make_logistic_word_model(random_state=random_state),
            param_grid={
                "classifier__C": list(LOGISTIC_TUNING_GRID["C"]),
                "classifier__class_weight": class_weights,
            },
            scoring=scorer,
            cv=inner_cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(texts[train_index], y_train)
        predictions = search.predict(texts[test_index])
        truth = labels[test_index]
        metrics = _fold_metric(truth, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predictions, labels=list(LABELS), zero_division=0
        )
        confusion += confusion_matrix(truth, predictions, labels=list(LABELS))
        for index, label in enumerate(LABELS):
            class_recalls[label].append(float(recall[index]))
        outer_rows.append(
            {
                "outer_fold": fold_number,
                "training_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                **metrics,
                **{
                    f"{label}_{metric}": float(values[index])
                    for metric, values in (("precision", precision), ("recall", recall), ("f1", f1), ("support", support))
                    for index, label in enumerate(LABELS)
                },
                "inner_best_macro_f1": float(search.best_score_),
                "selected_C": float(search.best_params_["classifier__C"]),
                "selected_class_weight": json.dumps(
                    _class_weight_for_metadata(search.best_params_["classifier__class_weight"]),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    outer_metric_rows = [
        {key: float(row[key]) for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}
        for row in outer_rows
    ]
    per_class_recall = {}
    for label, values in class_recalls.items():
        array = np.asarray(values, dtype=float)
        per_class_recall[label] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }

    # Choose final parameters using inner CV over all audited labels. This
    # score is a tuning score and must not be presented as held-out performance.
    full_inner_count = min(int(inner_splits), int(counts.min()))
    if full_inner_count < 2:
        raise ValueError("Không đủ mẫu để chọn hyperparameter cuối cùng.")
    full_search = GridSearchCV(
        estimator=_make_logistic_word_model(random_state=random_state),
        param_grid={
            "classifier__C": list(LOGISTIC_TUNING_GRID["C"]),
            "classifier__class_weight": class_weights,
        },
        scoring=scorer,
        cv=StratifiedKFold(
            n_splits=full_inner_count, shuffle=True, random_state=random_state + 1000
        ),
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    full_search.fit(texts, labels)
    full_params = {
        "C": float(full_search.best_params_["classifier__C"]),
        "class_weight": _class_weight_for_metadata(
            full_search.best_params_["classifier__class_weight"]
        ),
    }
    return {
        "protocol": "nested_stratified_cv",
        "outer_splits": int(outer_count),
        "outer_repeats": 1,
        "inner_splits": int(full_inner_count),
        "inner_selection_metric": "macro_f1",
        "outer_reporting_metrics": _summarize(outer_metric_rows),
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
        "outer_folds": outer_rows,
        "full_data_inner_best_params": full_params,
        "full_data_inner_best_macro_f1_tuning_score": float(full_search.best_score_),
        "final_model_name": "logistic_word_nested_tuned",
        "independent_test_available": False,
    }


def evaluate_models(
    data: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    nested_outer_splits: int = 5,
    nested_inner_splits: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Benchmark baseline and candidate models with repeated stratified CV."""
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Evaluation data must contain text and label columns.")
    texts = data["text"].fillna("").astype(str)
    labels = data["label"].fillna("").astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    if any(label not in counts or int(counts[label]) < 2 for label in LABELS):
        raise ValueError("Mỗi lớp cảm xúc cần ít nhất 2 mẫu để đánh giá.")
    n_splits = min(int(n_splits), int(counts.min()))
    if n_splits < 2:
        raise ValueError("Không đủ mẫu cho cross-validation.")
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=max(1, int(n_repeats)), random_state=random_state
    )
    results = [_evaluate_constant_baseline(labels, splitter)]
    for spec in MODEL_SPECS:
        results.append(
            _evaluate_spec(spec, texts, labels, splitter, random_state=random_state)
        )
    candidates = [result for result in results if result["name"] != "majority_baseline"]
    best = max(candidates, key=lambda result: (result["metrics"]["macro_f1_mean"], result["metrics"]["macro_recall_mean"]))
    nested_selection = evaluate_nested_logistic(
        data,
        outer_splits=nested_outer_splits,
        inner_splits=nested_inner_splits,
        random_state=random_state,
    )
    return {
        "protocol": "repeated_stratified_cv",
        "n_splits": int(n_splits),
        "n_repeats": int(max(1, int(n_repeats))),
        "folds_per_model": int(n_splits * max(1, int(n_repeats))),
        "random_state": int(random_state),
        "independent_test_available": False,
        "selection_metric": "macro_f1_mean",
        "best_model": best["name"],
        "nested_selection": nested_selection,
        "labels": list(LABELS),
        "models": results,
    }


def train_and_save_best_model(
    data: pd.DataFrame,
    evaluation: dict[str, Any],
    *,
    model_path: str | Path,
    metadata_path: str | Path,
    training_dataset_fingerprint: str,
    audit_summary: dict[str, Any] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit the nested-CV-selected LR on all audited rows and save metadata."""
    nested = evaluation.get("nested_selection")
    if not nested:
        raise ValueError("Evaluation is missing nested CV selection results.")
    selected_params = nested["full_data_inner_best_params"]
    class_weight = selected_params["class_weight"]
    model = _make_logistic_word_model(
        C=float(selected_params["C"]),
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(data["text"].fillna("").astype(str), data["label"].astype(str))
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "label_source": "manual",
        "training_samples": int(len(data)),
        "training_dataset_fingerprint": training_dataset_fingerprint,
        "model_name": nested["final_model_name"],
        "algorithm": "logistic_regression",
        "features": "word",
        "hyperparameters": {
            "C": float(selected_params["C"]),
            "class_weight": _class_weight_for_metadata(class_weight),
            "max_iter": 2000,
            "solver": "lbfgs",
        },
        "classes": list(LABELS),
        "evaluation_protocol": nested["protocol"],
        "nested_cross_validation": {
            "outer_splits": nested["outer_splits"],
            "outer_repeats": nested["outer_repeats"],
            "inner_splits": nested["inner_splits"],
            "selection_metric": nested["inner_selection_metric"],
            "outer_reporting_metrics": nested["outer_reporting_metrics"],
            "per_class_recall": nested["per_class_recall"],
        },
        "exploratory_comparison_protocol": evaluation["protocol"],
        "exploratory_best_model": evaluation["best_model"],
        "independent_test_available": False,
        "deployment_ready": False,
        "selection_metric": nested["inner_selection_metric"],
        "cross_validation_metrics": nested["outer_reporting_metrics"],
        "audit_summary": audit_summary or {},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def write_evaluation_artifacts(
    evaluation: dict[str, Any],
    *,
    output_dir: str | Path,
    audit_summary: dict[str, Any],
    issues: pd.DataFrame,
) -> None:
    """Write compact JSON/CSV artifacts for the dashboard and thesis report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined = {**audit_summary, **evaluation, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    (output / "summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for result in evaluation["models"]:
        row = {"model": result["name"], "algorithm": result["algorithm"], "features": result["features"]}
        row.update(result["metrics"])
        rows.append(row)
        for label, metrics in result.get("per_class", {}).items():
            per_class_rows.append({"model": result["name"], "label": label, **metrics})
        matrix = pd.DataFrame(result["confusion_matrix"], index=LABELS, columns=LABELS)
        matrix.to_csv(output / f"confusion_{result['name']}.csv", encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class_rows).to_csv(output / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    nested = evaluation.get("nested_selection")
    if nested:
        nested_summary = {
            key: value for key, value in nested.items() if key != "outer_folds"
        }
        (output / "nested_cv_summary.json").write_text(
            json.dumps(nested_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(nested["outer_folds"]).to_csv(
            output / "nested_cv_outer_folds.csv", index=False, encoding="utf-8-sig"
        )
        recall_rows = [
            {"label": label, **values}
            for label, values in nested["per_class_recall"].items()
        ]
        pd.DataFrame(recall_rows).to_csv(
            output / "nested_cv_per_class_recall.csv", index=False, encoding="utf-8-sig"
        )
    issues.to_csv(output / "audit_issues.csv", index=False, encoding="utf-8-sig")
