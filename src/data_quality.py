"""Audits human sentiment labels without modifying their source files.

This module deliberately keeps proposed/audited rows separate from source CSVs.
Only rows explicitly marked ``sentiment_source == 'manual'`` are eligible for
model training. Historical review predictions are retained as evidence, never
promoted to gold labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable

import pandas as pd


SENTIMENT_MAP = {
    "tiêu cực": "negative",
    "trung tính": "neutral",
    "tích cực": "positive",
}
TRAIN_COLUMNS = ("sentiment_manual", "text")


def normalize_text(value: Any) -> str:
    """Normalize Unicode, case and whitespace for identity comparisons only."""
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def text_fingerprint(value: Any) -> str:
    normalized = normalize_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _sentiment(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return SENTIMENT_MAP.get(normalize_text(value), "")


def _text_for_row(row: pd.Series, preferred: Iterable[str]) -> str:
    for column in preferred:
        value = row.get(column)
        normalized = normalize_text(value)
        if normalized:
            return str(value)
    return ""


def _reference_platforms(processed: pd.DataFrame | None) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    if processed is None or "review_id" not in processed.columns or "platform" not in processed.columns:
        return result
    for _, row in processed.iterrows():
        review_id = normalize_text(row.get("review_id"))
        platform = normalize_text(row.get("platform"))
        if review_id and platform:
            result[review_id].add(platform)
    return result


def _processed_lookup(processed: pd.DataFrame | None) -> dict[tuple[str, str], list[dict[str, str]]]:
    lookup: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    if processed is None or "review_id" not in processed.columns:
        return lookup
    for _, row in processed.iterrows():
        review_id = normalize_text(row.get("review_id"))
        if not review_id:
            continue
        raw_text = _text_for_row(row, ("comment_text_raw", "comment_text", "comment_text_clean"))
        lookup[(review_id, text_fingerprint(raw_text))].append({
            "platform": normalize_text(row.get("platform")),
            "product_name": str(row.get("product_name", "") or ""),
            "category": str(row.get("category", "") or ""),
            "rating": str(row.get("rating", "") or ""),
            "comment_text_raw": raw_text,
        })
    return lookup


def _add_identity_columns(
    frame: pd.DataFrame,
    *,
    text_columns: tuple[str, ...],
    processed: pd.DataFrame | None,
    source_name: str,
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["source_file"] = source_name
    result["source_row"] = range(1, len(result) + 1)
    platforms_by_id = _reference_platforms(processed)
    processed_lookup = _processed_lookup(processed)

    canonical_texts: list[str] = []
    platforms: list[str] = []
    keys: list[str] = []
    fingerprints: list[str] = []
    identity_issues: list[str] = []
    evidence_rows: list[dict[str, str]] = []

    for _, row in result.iterrows():
        text = _text_for_row(row, text_columns)
        review_id_value = row.get("review_id")
        review_id = normalize_text(review_id_value)
        fingerprint = text_fingerprint(text)
        platform = normalize_text(row.get("platform"))
        identity_issue: list[str] = []

        candidates = processed_lookup.get((review_id, fingerprint), []) if review_id and fingerprint else []
        if not platform and candidates:
            candidate_platforms = {candidate["platform"] for candidate in candidates if candidate["platform"]}
            if len(candidate_platforms) == 1:
                platform = next(iter(candidate_platforms))
            elif len(candidate_platforms) > 1:
                identity_issue.append("ambiguous_platform_match")
        if not platform and review_id:
            known_platforms = platforms_by_id.get(review_id, set())
            if len(known_platforms) == 1:
                platform = next(iter(known_platforms))
            elif len(known_platforms) > 1:
                identity_issue.append("ambiguous_platform_match")

        if review_id and platform:
            key = f"{platform}::{review_id}"
        elif review_id:
            key = f"unknown::{review_id}"
            if not identity_issue:
                identity_issue.append("platform_unresolved")
        elif fingerprint:
            key = f"text::{fingerprint}"
            identity_issue.append("missing_review_id")
        else:
            key = f"row::{source_name}::{len(keys) + 1}"
            identity_issue.append("missing_review_id_and_text")

        canonical_texts.append(text)
        platforms.append(platform)
        keys.append(key)
        fingerprints.append(fingerprint)
        identity_issues.append(";".join(identity_issue))
        evidence_rows.append(candidates[0] if len(candidates) == 1 else {})

    result["canonical_text"] = canonical_texts
    result["canonical_platform"] = platforms
    result["review_key"] = keys
    result["text_fingerprint"] = fingerprints
    result["identity_issues"] = identity_issues
    result["processed_evidence"] = evidence_rows
    return result


def _make_issue(row: pd.Series, code: str, detail: str = "") -> dict[str, Any]:
    return {
        "issue_code": code,
        "detail": detail,
        "source_file": row.get("source_file", ""),
        "source_row": row.get("source_row", ""),
        "review_id": row.get("review_id", ""),
        "platform": row.get("canonical_platform", row.get("platform", "")),
        "review_key": row.get("review_key", ""),
        "text_fingerprint": row.get("text_fingerprint", ""),
        "sentiment_manual": row.get("sentiment_manual", ""),
        "canonical_text": row.get("canonical_text", ""),
    }


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _audit_training_rows(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = training.copy().reset_index(drop=True)
    issues: list[dict[str, Any]] = []
    statuses = ["usable"] * len(result)
    issue_strings = [""] * len(result)
    label_values = [_sentiment(value) for value in result.get("sentiment_manual", pd.Series(index=result.index, dtype=object))]
    result["sentiment_label"] = label_values

    for index, row in result.iterrows():
        row_codes: list[str] = []
        if not label_values[index]:
            row_codes.append("invalid_or_missing_manual_sentiment")
        if not normalize_text(row.get("canonical_text")):
            row_codes.append("empty_training_text")
        if str(row.get("identity_issues", "")):
            identity_codes = str(row.get("identity_issues")).split(";")
            row_codes.extend(identity_codes)
        else:
            identity_codes = []
        # Unresolved platform or ID alone is an audit warning: matching by
        # normalized text still prevents leakage/duplicate rows. An ambiguous
        # platform match or a row lacking both ID and text is unsafe.
        blocking_codes = {"ambiguous_platform_match", "missing_review_id_and_text"}
        if any(code in blocking_codes for code in row_codes) or any(
            code in {"invalid_or_missing_manual_sentiment", "empty_training_text"}
            for code in row_codes
        ):
            statuses[index] = "quarantined"
        issue_strings[index] = ";".join(sorted(set(row_codes)))
        for code in sorted(set(row_codes)):
            issues.append(_make_issue(row, code))

    # Link rows sharing either a platform+review ID or normalized text. Any
    # contradictory component is quarantined in full, including transitive links.
    union_find = _UnionFind(len(result))
    for identity_column in ("review_key", "text_fingerprint"):
        first: dict[str, int] = {}
        for index, value in enumerate(result[identity_column].fillna("")):
            value = str(value)
            if not value or value.startswith("row::"):
                continue
            if value in first:
                union_find.union(index, first[value])
            else:
                first[value] = index

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(result)):
        components[union_find.find(index)].append(index)

    keep_indices: list[int] = []
    for component in components.values():
        labels = {label_values[index] for index in component if label_values[index]}
        component_keys = {str(result.iloc[index]["review_key"]) for index in component}
        key_texts: dict[str, set[str]] = defaultdict(set)
        for index in component:
            key = str(result.iloc[index]["review_key"])
            fingerprint = str(result.iloc[index]["text_fingerprint"])
            if fingerprint:
                key_texts[key].add(fingerprint)

        conflict = len(labels) > 1 or any(len(fingerprints) > 1 for fingerprints in key_texts.values())
        if conflict:
            for index in component:
                statuses[index] = "quarantined"
                codes = set(filter(None, issue_strings[index].split(";")))
                codes.add("conflicting_connected_labels_or_text")
                issue_strings[index] = ";".join(sorted(codes))
                issues.append(_make_issue(result.iloc[index], "conflicting_connected_labels_or_text"))
            continue

        # Rows already invalid remain quarantined. Otherwise retain the first
        # equivalent record and flag later identical rows as consistent duplicates.
        candidates = [index for index in component if statuses[index] == "usable"]
        if not candidates:
            continue
        keep = min(candidates)
        keep_indices.append(keep)
        for index in candidates:
            if index == keep:
                continue
            statuses[index] = "duplicate_consistent"
            issue_strings[index] = "duplicate_consistent"
            issues.append(_make_issue(result.iloc[index], "duplicate_consistent", "Equivalent duplicate; first source row retained."))

    result["audit_status"] = statuses
    result["audit_issues"] = issue_strings
    # The prepared training set contains only the deterministic representative
    # of each consistent group. Source CSV records stay untouched.
    usable = result.loc[keep_indices].copy().reset_index(drop=True)
    return usable, pd.DataFrame(issues)


def prepare_training_labels(
    train: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    processed: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return clean training representatives plus row-level audit issues.

    Args:
        train: Manual-label CSV contents. Only explicit manual sources qualify.
        reference: Optional legacy evaluation/candidate data; used to report
            overlap and conflicts, never to add labels to training.
        processed: Optional processed reviews, used to resolve platform for
            older label rows that have no platform column.
    """
    missing = set(TRAIN_COLUMNS).difference(train.columns)
    if missing:
        raise ValueError(f"Training labels missing required columns: {', '.join(sorted(missing))}")
    if "sentiment_source" in train.columns:
        eligible = train[train["sentiment_source"].fillna("").astype(str).str.strip().str.casefold().eq("manual")].copy()
    else:
        eligible = train.iloc[0:0].copy()

    training = _add_identity_columns(
        eligible,
        text_columns=("text", "text_model"),
        processed=processed,
        source_name="manual_labels.csv",
    )
    usable, issues = _audit_training_rows(training)

    if reference is not None and not reference.empty:
        reference_frame = _add_identity_columns(
            reference,
            text_columns=("comment_text_raw", "text", "comment_text_segmented", "text_model"),
            processed=processed,
            source_name="manual_labels_test.csv",
        )
        usable_keys = set(usable["review_key"].astype(str))
        usable_fingerprints = set(usable["text_fingerprint"].astype(str)) - {""}
        ref_labels = [_sentiment(value) for value in reference_frame.get("sentiment_manual", pd.Series(index=reference_frame.index, dtype=object))]
        additional: list[dict[str, Any]] = []
        for index, row in reference_frame.iterrows():
            key_overlap = row["review_key"] in usable_keys
            text_overlap = bool(row["text_fingerprint"] and row["text_fingerprint"] in usable_fingerprints)
            if key_overlap or text_overlap:
                issue_code = "legacy_reference_overlaps_training"
                if ref_labels[index]:
                    issue_code += "_not_independent_test"
                additional.append(_make_issue(row, issue_code))
        if additional:
            issues = pd.concat([issues, pd.DataFrame(additional)], ignore_index=True)

        # A legacy reference's manually entered label is evidence for review,
        # not a training target. If it disagrees with the training label, the
        # matching training example must be quarantined until adjudicated.
        train_by_key: dict[str, list[int]] = defaultdict(list)
        train_by_text: dict[str, list[int]] = defaultdict(list)
        for i, row in training.iterrows():
            if _sentiment(row.get("sentiment_manual")):
                train_by_key[str(row["review_key"])].append(i)
                if row["text_fingerprint"]:
                    train_by_text[str(row["text_fingerprint"])].append(i)
        disagreements: set[int] = set()
        for i, row in reference_frame.iterrows():
            reference_label = ref_labels[i]
            if not reference_label:
                continue
            matching = set(train_by_key.get(str(row["review_key"]), []))
            if row["text_fingerprint"]:
                matching.update(train_by_text.get(str(row["text_fingerprint"]), []))
            for train_index in matching:
                if _sentiment(training.iloc[train_index].get("sentiment_manual")) != reference_label:
                    disagreements.add(train_index)
                    issues = pd.concat([issues, pd.DataFrame([
                        _make_issue(training.iloc[train_index], "reference_label_conflict", f"Legacy reference row {row['source_row']} has a different manually entered sentiment."),
                        _make_issue(row, "reference_label_conflict", f"Training source row {training.iloc[train_index]['source_row']} has a different manually entered sentiment."),
                    ])], ignore_index=True)
        if disagreements:
            conflict_rows = training.iloc[sorted(disagreements)]
            conflict_keys = set(conflict_rows["review_key"].astype(str))
            conflict_texts = set(conflict_rows["text_fingerprint"].astype(str)) - {""}
            # Rebuild the training set with labels excluded when a legacy label
            # conflict is unresolved. Keep these rows in the audit artifact.
            unresolved = training[
                training["review_key"].astype(str).isin(conflict_keys)
                | training["text_fingerprint"].astype(str).isin(conflict_texts)
            ].copy()
            for _, row in unresolved.iterrows():
                issues = pd.concat([issues, pd.DataFrame([_make_issue(row, "reference_label_conflict")])], ignore_index=True)
            blocked = set(unresolved["source_row"].astype(int))
            usable = usable[~usable["source_row"].astype(int).isin(blocked)].copy().reset_index(drop=True)

    return usable, issues.reset_index(drop=True)


def audit_label_datasets(
    train: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    processed: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Prepare report tables and a summary for the read-only audit CLI."""
    usable, issues = prepare_training_labels(train, reference, processed)
    labeled = train.copy().reset_index(drop=True)
    labeled["source_file"] = "manual_labels.csv"
    labeled["source_row"] = range(1, len(labeled) + 1)

    # Recreate all train rows with statuses for explicit reporting, including
    # excluded non-manual rows (these are never silently used for training).
    if "sentiment_source" in labeled.columns:
        manual_mask = labeled["sentiment_source"].fillna("").astype(str).str.strip().str.casefold().eq("manual")
    else:
        manual_mask = pd.Series(False, index=labeled.index)
    manual_labeled = labeled[manual_mask].copy()
    identity_manual = _add_identity_columns(manual_labeled, text_columns=("text", "text_model"), processed=processed, source_name="manual_labels.csv")
    training_audit = identity_manual.copy()
    training_audit["sentiment_label"] = training_audit.get("sentiment_manual", pd.Series(index=training_audit.index, dtype=object)).map(_sentiment)
    train_issues = issues[issues.get("source_file", pd.Series(index=issues.index, dtype=object)).eq("manual_labels.csv")].copy() if not issues.empty else pd.DataFrame()
    issue_codes_by_row: dict[int, list[str]] = defaultdict(list)
    for _, issue in train_issues.iterrows():
        try:
            issue_row = int(issue["source_row"])
        except (TypeError, ValueError):
            continue
        issue_codes_by_row[issue_row].append(str(issue["issue_code"]))
    usable_source_rows = set(usable["source_row"].astype(int)) if not usable.empty else set()
    statuses: list[str] = []
    audit_issue_strings: list[str] = []
    for _, row in training_audit.iterrows():
        source_row = int(row["source_row"])
        codes = sorted(set(issue_codes_by_row.get(source_row, [])))
        if source_row in usable_source_rows:
            status = "usable"
        elif "duplicate_consistent" in codes:
            status = "duplicate_consistent"
        else:
            status = "quarantined"
        statuses.append(status)
        audit_issue_strings.append(";".join(codes))
    training_audit["audit_status"] = statuses
    training_audit["audit_issues"] = audit_issue_strings

    issue_rows: list[dict[str, Any]] = [row.to_dict() for _, row in issues.iterrows()]
    # Report and mark source-only non-manual records without turning them into gold labels.
    for _, row in labeled[~manual_mask].iterrows():
        issue_rows.append({
            "issue_code": "not_explicitly_manual_source",
            "detail": "Excluded from training; source is not marked manual.",
            "source_file": "manual_labels.csv",
            "source_row": int(row["source_row"]),
            "review_id": row.get("review_id", ""),
            "sentiment_manual": row.get("sentiment_manual", ""),
        })
    issues = pd.DataFrame(issue_rows)

    annotation_candidates = pd.DataFrame()
    reference_rows = pd.DataFrame()
    if reference is not None:
        reference_rows = _add_identity_columns(
            reference,
            text_columns=("comment_text_raw", "text", "comment_text_segmented", "text_model"),
            processed=processed,
            source_name="manual_labels_test.csv",
        )

    if not reference_rows.empty:
        ref_copy = reference_rows.copy()
        all_train_key_set = set(identity_manual["review_key"].astype(str))
        all_train_fp_set = set(identity_manual["text_fingerprint"].astype(str)) - {""}
        ref_copy["sentiment_label"] = ref_copy.get("sentiment_manual", pd.Series(index=ref_copy.index, dtype=object)).map(_sentiment)
        ref_copy["candidate_provenance"] = "manual_labels_test.csv:legacy_review_queue"
        ref_copy["human_confirmation_required"] = True
        ref_copy["legacy_test_overlap"] = ref_copy["review_key"].astype(str).isin(all_train_key_set) | ref_copy["text_fingerprint"].astype(str).isin(all_train_fp_set)
        # The file includes AI/reviewer fields; preserve them as provenance only.
        annotation_candidates = ref_copy.copy()

    # Side-by-side queue: retain all legacy rows and attach every matching
    # manual training record as evidence. Add train-only rows at the end so no
    # source record disappears from review.
    training_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    training_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in training_audit.iterrows():
        record = row.to_dict()
        training_by_key[str(row["review_key"])].append(record)
        if row["text_fingerprint"]:
            training_by_text[str(row["text_fingerprint"])].append(record)

    queue_rows: list[dict[str, Any]] = []
    matched_training_rows: set[int] = set()
    if not reference_rows.empty:
        for _, ref_row in annotation_candidates.iterrows():
            matches: dict[int, dict[str, Any]] = {}
            for record in training_by_key.get(str(ref_row["review_key"]), []):
                matches[int(record["source_row"])] = record
            if ref_row["text_fingerprint"]:
                for record in training_by_text.get(str(ref_row["text_fingerprint"]), []):
                    matches[int(record["source_row"])] = record
            matched_training_rows.update(matches)
            ordered_matches = [matches[key] for key in sorted(matches)]
            train_record = ordered_matches[0] if ordered_matches else {}
            queue_rows.append({
                "queue_source": "legacy_reference",
                "human_confirmation_required": True,
                "legacy_test_overlap": bool(ordered_matches),
                "review_key": ref_row.get("review_key", ""),
                "text_fingerprint": ref_row.get("text_fingerprint", ""),
                "reference_source_row": ref_row.get("source_row", ""),
                "reference_review_id": ref_row.get("review_id", ""),
                "reference_platform": ref_row.get("canonical_platform", ref_row.get("platform", "")),
                "reference_text_raw": ref_row.get("comment_text_raw", ref_row.get("text", "")),
                "reference_text_segmented": ref_row.get("comment_text_segmented", ""),
                "reference_sentiment_manual": ref_row.get("sentiment_manual", ""),
                "reference_aspects_manual": ref_row.get("aspects_manual", ""),
                "reference_sentiment_review": ref_row.get("sentiment_review", ""),
                "reference_aspects_review": ref_row.get("aspects_review", ""),
                "reference_ai_review_status": ref_row.get("ai_review_status", ""),
                "reference_annotation_status": ref_row.get("annotation_status", ""),
                "reference_labelled_at": ref_row.get("labelled_at", ""),
                "train_match_count": len(ordered_matches),
                "train_source_rows": ",".join(str(row["source_row"]) for row in ordered_matches),
                "train_review_id": train_record.get("review_id", ""),
                "train_platform": train_record.get("canonical_platform", ""),
                "train_text_raw": train_record.get("text", ""),
                "train_text_model": train_record.get("text_model", ""),
                "train_sentiment_manual": train_record.get("sentiment_manual", ""),
                "train_aspects_manual": train_record.get("aspects_manual", ""),
                "train_labelled_at": train_record.get("labelled_at", ""),
                "train_audit_status": train_record.get("audit_status", ""),
                "train_audit_issues": train_record.get("audit_issues", ""),
            })

    for _, train_row in training_audit.iterrows():
        source_row = int(train_row["source_row"])
        if source_row in matched_training_rows:
            continue
        queue_rows.append({
            "queue_source": "training_only",
            "human_confirmation_required": train_row["audit_status"] != "usable",
            "legacy_test_overlap": False,
            "review_key": train_row.get("review_key", ""),
            "text_fingerprint": train_row.get("text_fingerprint", ""),
            "reference_source_row": "",
            "reference_review_id": "",
            "reference_platform": "",
            "reference_text_raw": "",
            "reference_text_segmented": "",
            "reference_sentiment_manual": "",
            "reference_aspects_manual": "",
            "reference_sentiment_review": "",
            "reference_aspects_review": "",
            "reference_ai_review_status": "",
            "reference_annotation_status": "",
            "reference_labelled_at": "",
            "train_match_count": 1,
            "train_source_rows": str(source_row),
            "train_review_id": train_row.get("review_id", ""),
            "train_platform": train_row.get("canonical_platform", ""),
            "train_text_raw": train_row.get("text", ""),
            "train_text_model": train_row.get("text_model", ""),
            "train_sentiment_manual": train_row.get("sentiment_manual", ""),
            "train_aspects_manual": train_row.get("aspects_manual", ""),
            "train_labelled_at": train_row.get("labelled_at", ""),
            "train_audit_status": train_row.get("audit_status", ""),
            "train_audit_issues": train_row.get("audit_issues", ""),
        })
    review_queue = pd.DataFrame(queue_rows)

    train_issue_codes = issues[
        issues.get("source_file", pd.Series(index=issues.index, dtype=object)).eq("manual_labels.csv")
    ]["issue_code"].value_counts().to_dict() if not issues.empty and "issue_code" in issues else {}
    training_quarantined = int(training_audit["audit_status"].eq("quarantined").sum()) if not training_audit.empty else 0
    summary = {
        "training_source_rows": int(len(train)),
        "manual_source_rows": int(manual_mask.sum()),
        "non_manual_rows_excluded": int((~manual_mask).sum()),
        "usable_training_rows": int(len(usable)),
        "training_issue_rows": int(sum(train_issue_codes.values())),
        "training_quarantined_rows": training_quarantined,
        "audit_issue_rows": int(len(issues)),
        "training_issue_counts": {str(key): int(value) for key, value in train_issue_codes.items()},
        "legacy_reference_rows": int(len(reference)) if reference is not None else 0,
        "legacy_reference_labeled_rows": int(reference["sentiment_manual"].notna().sum()) if reference is not None and "sentiment_manual" in reference else 0,
        "legacy_reference_unlabeled_rows": int(reference["sentiment_manual"].isna().sum()) if reference is not None and "sentiment_manual" in reference else int(len(reference)) if reference is not None else 0,
        "legacy_reference_ai_reviewed_rows": int(reference["ai_review_status"].eq("completed").sum()) if reference is not None and "ai_review_status" in reference else 0,
        "legacy_reference_is_independent_test": False,
        "training_label_distribution": {str(key): int(value) for key, value in usable["sentiment_label"].value_counts().items()} if "sentiment_label" in usable else {},
        "training_dataset_fingerprint": dataset_fingerprint(train),
        "reference_dataset_fingerprint": dataset_fingerprint(reference) if reference is not None else None,
        "processed_dataset_fingerprint": dataset_fingerprint(processed) if processed is not None else None,
    }
    return {
        "summary": summary,
        "issues": issues.reset_index(drop=True),
        "training_labels": usable.reset_index(drop=True),
        "training_audit": training_audit.reset_index(drop=True),
        "review_queue": review_queue.reset_index(drop=True),
        "annotation_candidates": annotation_candidates.reset_index(drop=True),
    }


def dataset_fingerprint(*frames: pd.DataFrame | None) -> str:
    """Stable SHA-256 over table content, independent of DataFrame index/order."""
    digest = hashlib.sha256()
    for frame in frames:
        if frame is None:
            digest.update(b"<none>\n")
            continue
        digest.update(json.dumps([str(column) for column in frame.columns], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
        rows: list[str] = []
        for values in frame.itertuples(index=False, name=None):
            record = {}
            for column, value in zip(frame.columns, values):
                if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
                    record[str(column)] = None
                elif isinstance(value, (list, dict, tuple)):
                    record[str(column)] = value
                else:
                    record[str(column)] = str(value)
            rows.append(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str))
        for row in sorted(rows):
            digest.update(row.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()
