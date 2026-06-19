# -*- coding: utf-8 -*-
"""
IEEE Access robustness experiments for human-authored knowledge injection.

This runner keeps the main contribution narrow, but adds the evidence reviewers
are likely to ask for:

1. threshold / same-project-floor sensitivity
2. multi-subset robustness over balanced 120-record samples
3. human-authored knowledge injection ablations
4. synthetic-vs-profile distribution similarity
5. simple classic baseline sanity checks

It deliberately reuses the previous JIPS Project Signature code and the current
human-authored knowledge injection logic instead of introducing a new method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.extraction_pipeline.create_external_test_subset import (
    choose_subset,
    summarize as summarize_subset,
    to_compatible_record,
)
from experiments.knowledge_injection.run_human_authored_injection_experiment import (
    DEFAULT_DATASET,
    DEFAULT_KNOWLEDGE_DIR,
    DEFAULT_MODE,
    DEFAULT_THRESHOLD,
    canonical_pair,
    case_pair_counter,
    comparison_delta,
    condition_result,
    is_generic_short_mention,
    is_specific_alias,
    item_text,
    lexical_score,
    norm,
    parse_human_authored_knowledge,
    read_jsonl,
    write_json,
    write_labels_jsonl,
)
from experiments.knowledge_injection.run_human_authored_injection_experiment import (  # noqa: E402
    _UnionFind,
    _cluster_merge_is_consistent,
    _has_conflict,
    _prepare_items,
    _records_to_items,
    _score_pair,
    cluster_items,
    evaluate_cluster_labels,
)


DEFAULT_SOURCE_MENTIONS = r"data\synthetic\extraction_run_20260618_085129\mentions_extracted.jsonl"
DEFAULT_SYNTHETIC_EMAILS = r"data\synthetic\synthetic_emails.jsonl"
DEFAULT_PROFILE_BINS = r"data\profiles\distribution_bins.csv"
DEFAULT_OUT_DIR = r"data\synthetic\ieee_access_runs\run_{timestamp}"
DEFAULT_SUBSET_SEEDS = [20260618, 20260619, 20260620, 20260621, 20260622]
DEFAULT_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]
DEFAULT_FLOORS = [0.90, 0.94, 0.96, 0.98]
DEFAULT_SUBSET_SIZE = 120

METRIC_KEYS = [
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "bcubed_precision",
    "bcubed_recall",
    "bcubed_f1",
    "cluster_purity",
    "false_merge_count",
    "false_split_count",
    "pred_cluster_count",
]


@dataclass
class PreparedExperiment:
    records: list[dict[str, Any]]
    items: list[dict[str, Any]]
    gold_labels: list[str]
    base_scores: dict[tuple[int, int], float]
    base_conflicts: dict[tuple[int, int], bool]
    mode: str


@dataclass
class MdPairInfo:
    relation: str
    a_project: str = ""
    b_project: str = ""
    a_confidence: float = 0.0
    b_confidence: float = 0.0
    a_evidence: str = ""
    b_evidence: str = ""


@dataclass
class MdPolicy:
    use_cannot_merge: bool = True
    use_same_project_floor: bool = True
    use_weak_boost: bool = True
    generic_safety: bool = True
    weak_boost: float = 0.08
    weak_cap: float = 0.80


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_number_list(value: str, *, as_int: bool = False) -> list[int] | list[float]:
    out = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part) if as_int else float(part))
    return out


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "condition",
        "role",
        "threshold",
        "same_project_floor",
        "ablation",
        "seed",
        "record_count",
        "gold_cluster_count",
        "pred_cluster_count",
        "pairwise_precision",
        "pairwise_recall",
        "pairwise_f1",
        "bcubed_precision",
        "bcubed_recall",
        "bcubed_f1",
        "cluster_purity",
        "false_merge_count",
        "false_split_count",
        "true_positive_pairs",
        "true_negative_pairs",
    ]
    return {key: row.get(key) for key in keys if key in row}


def prepare_experiment(records: list[dict[str, Any]], mode: str) -> PreparedExperiment:
    items = _records_to_items(records)
    prepared = _prepare_items(items, [mode])
    base_scores, base_conflicts = precompute_base_pairs(prepared, mode)
    return PreparedExperiment(
        records=records,
        items=prepared,
        gold_labels=[item["gold_cluster_id"] for item in prepared],
        base_scores=base_scores,
        base_conflicts=base_conflicts,
        mode=mode,
    )


def precompute_base_pairs(
    items: list[dict[str, Any]],
    mode: str,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], bool]]:
    scores: dict[tuple[int, int], float] = {}
    conflicts: dict[tuple[int, int], bool] = {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            key = (i, j)
            scores[key] = _score_pair(items[i], items[j], mode)
            conflicts[key] = _has_conflict(items[i], items[j], mode)
    return scores, conflicts


def match_item_to_kb_policy(
    item: dict[str, Any],
    kb,
    *,
    generic_safety: bool,
) -> tuple[str, float, str]:
    mention = str(item.get("mention") or "")
    mention_key = norm(mention)
    generic_mention = is_generic_short_mention(mention) if generic_safety else False
    text = item_text(item)
    text_key = norm(text)

    best_canonical = ""
    best_score = 0.0
    best_evidence = ""

    def update(canonical: str, score: float, evidence: str) -> None:
        nonlocal best_canonical, best_score, best_evidence
        if score > best_score:
            best_canonical = canonical
            best_score = score
            best_evidence = evidence

    if mention_key in kb.alias_to_canonical:
        update(kb.alias_to_canonical[mention_key], 1.0, mention)

    for project in kb.projects.values():
        aliases = [project.canonical, *project.aliases]
        for alias in aliases:
            alias_key = norm(alias)
            if not alias_key:
                continue
            if mention_key == alias_key:
                update(project.canonical, 1.0, alias)
            elif is_specific_alias(alias) and (
                alias_key in mention_key or (not generic_safety and mention_key and mention_key in alias_key)
            ):
                update(project.canonical, 0.92, alias)
            elif is_specific_alias(alias) and alias_key in text_key:
                update(project.canonical, 0.86, alias)
            elif not generic_mention:
                sim = lexical_score(mention, alias)
                if sim >= 0.82:
                    update(project.canonical, 0.80 + min(0.12, (sim - 0.82) / 2), alias)

        weak_hits = 0
        if project.system and project.system.lower() in text.lower():
            weak_hits += 1
        if any(module and module in text for module in project.modules):
            weak_hits += 1
        if any(work and work in text for work in project.work_types):
            weak_hits += 1
        if weak_hits >= 3:
            update(project.canonical, 0.72, "system+module+work_type")

    if best_score < 0.72:
        return "", best_score, best_evidence
    return best_canonical, best_score, best_evidence


def precompute_md_pairs(
    items: list[dict[str, Any]],
    kb,
    *,
    generic_safety: bool,
) -> tuple[dict[tuple[int, int], MdPairInfo], Counter]:
    matches = [
        match_item_to_kb_policy(item, kb, generic_safety=generic_safety)
        for item in items
    ]
    pairs: dict[tuple[int, int], MdPairInfo] = {}
    counts = Counter()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_project, a_conf, a_evidence = matches[i]
            b_project, b_conf, b_evidence = matches[j]
            relation = "baseline"
            if a_project and b_project:
                if canonical_pair(a_project, b_project) in kb.cannot_pairs:
                    relation = "md_cannot_merge"
                elif norm(a_project) == norm(b_project):
                    min_conf = min(a_conf, b_conf)
                    if min_conf >= 0.90:
                        relation = "md_same_project"
                    elif min_conf >= 0.82:
                        relation = "md_weak_same_project"
            counts[relation] += 1
            pairs[(i, j)] = MdPairInfo(
                relation=relation,
                a_project=a_project,
                b_project=b_project,
                a_confidence=a_conf,
                b_confidence=b_conf,
                a_evidence=a_evidence,
                b_evidence=b_evidence,
            )
    return pairs, counts


def build_md_pair_data(
    prepared: PreparedExperiment,
    md_pairs: dict[tuple[int, int], MdPairInfo],
    *,
    policy: MdPolicy,
    same_project_floor: float,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], bool], Counter]:
    scores: dict[tuple[int, int], float] = {}
    conflicts: dict[tuple[int, int], bool] = {}
    reasons = Counter()
    for key, base_score in prepared.base_scores.items():
        info = md_pairs[key]
        score = base_score
        conflict = prepared.base_conflicts[key]
        reason = "baseline"
        if info.relation == "md_cannot_merge" and policy.use_cannot_merge:
            score = 0.0
            conflict = True
            reason = "md_cannot_merge"
        elif info.relation == "md_same_project" and policy.use_same_project_floor:
            score = max(base_score, same_project_floor)
            reason = "md_same_project"
        elif info.relation == "md_weak_same_project" and policy.use_weak_boost:
            score = max(base_score, min(policy.weak_cap, base_score + policy.weak_boost))
            reason = "md_weak_same_project"
        scores[key] = max(0.0, min(1.0, score))
        conflicts[key] = bool(conflict)
        reasons[reason] += 1
    return scores, conflicts, reasons


def cluster_from_pair_data(
    n: int,
    pair_scores: dict[tuple[int, int], float],
    pair_conflicts: dict[tuple[int, int], bool],
    *,
    threshold: float,
    label_prefix: str,
) -> list[str]:
    uf = _UnionFind(n)
    ranked = sorted(pair_scores.items(), key=lambda kv: -kv[1])
    review_threshold = max(0.0, threshold - 0.12)

    for (i, j), score in ranked:
        if score < threshold:
            break
        if pair_conflicts.get((i, j)):
            continue
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if not _cluster_merge_is_consistent(uf, ri, rj, pair_scores, pair_conflicts, review_threshold):
            continue
        uf.union(ri, rj)

    root_to_label: dict[int, str] = {}
    labels = []
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_label:
            root_to_label[root] = f"{label_prefix}{len(root_to_label) + 1:04d}"
        labels.append(root_to_label[root])
    return labels


def evaluate_prepared_baseline(
    prepared: PreparedExperiment,
    *,
    threshold: float,
    name: str = "baseline_signature",
) -> tuple[dict[str, Any], list[str]]:
    # Keep the baseline definition identical to the previous JIPS implementation.
    labels = cluster_items(prepared.items, prepared.mode, threshold)
    row = condition_result(
        name=name,
        role="baseline",
        items=prepared.items,
        gold_labels=prepared.gold_labels,
        labels=labels,
        threshold=threshold,
    )
    return row, labels


def evaluate_prepared_md(
    prepared: PreparedExperiment,
    *,
    condition: str,
    kb,
    threshold: float,
    same_project_floor: float,
    policy: MdPolicy,
    md_pairs_cache: dict[str, tuple[dict[tuple[int, int], MdPairInfo], Counter]],
) -> tuple[dict[str, Any], list[str]]:
    cache_key = "safe" if policy.generic_safety else "unsafe_generic"
    if cache_key not in md_pairs_cache:
        md_pairs_cache[cache_key] = precompute_md_pairs(
            prepared.items,
            kb,
            generic_safety=policy.generic_safety,
        )
    md_pairs, raw_relation_counts = md_pairs_cache[cache_key]
    pair_scores, pair_conflicts, applied_counts = build_md_pair_data(
        prepared,
        md_pairs,
        policy=policy,
        same_project_floor=same_project_floor,
    )
    labels = cluster_from_pair_data(
        len(prepared.items),
        pair_scores,
        pair_conflicts,
        threshold=threshold,
        label_prefix="MD",
    )
    debug = {
        "knowledge_file": getattr(kb, "path", ""),
        "knowledge_project_count": len(kb.projects),
        "knowledge_alias_count": len(kb.alias_to_canonical),
        "knowledge_cannot_pair_count": len(kb.cannot_pairs),
        "raw_relation_counts": dict(raw_relation_counts),
        "applied_adjustment_counts": dict(applied_counts),
        "policy": {
            "use_cannot_merge": policy.use_cannot_merge,
            "use_same_project_floor": policy.use_same_project_floor,
            "use_weak_boost": policy.use_weak_boost,
            "generic_safety": policy.generic_safety,
        },
    }
    row = condition_result(
        name=condition,
        role="human_authored_knowledge",
        items=prepared.items,
        gold_labels=prepared.gold_labels,
        labels=labels,
        threshold=threshold,
        debug=debug,
    )
    row["same_project_floor"] = same_project_floor
    return row, labels


def evaluate_main_conditions(
    records: list[dict[str, Any]],
    *,
    knowledge_dir: Path,
    mode: str,
    threshold: float,
    same_project_floor: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    prepared = prepare_experiment(records, mode)
    rows = []
    labels_by_condition = {}
    baseline, labels = evaluate_prepared_baseline(prepared, threshold=threshold)
    baseline["same_project_floor"] = same_project_floor
    rows.append(baseline)
    labels_by_condition["baseline_signature"] = labels

    for condition, filename in [
        ("human_partial", "project_knowledge_partial.md"),
        ("human_full", "project_knowledge_full.md"),
        ("human_noisy", "project_knowledge_noisy.md"),
    ]:
        kb = parse_human_authored_knowledge(knowledge_dir / filename)
        md_row, md_labels = evaluate_prepared_md(
            prepared,
            condition=condition,
            kb=kb,
            threshold=threshold,
            same_project_floor=same_project_floor,
            policy=MdPolicy(),
            md_pairs_cache={},
        )
        rows.append(md_row)
        labels_by_condition[condition] = md_labels
    return rows, labels_by_condition


def run_threshold_sweep(
    records: list[dict[str, Any]],
    *,
    knowledge_dir: Path,
    mode: str,
    thresholds: list[float],
    same_project_floor: float,
) -> list[dict[str, Any]]:
    prepared = prepare_experiment(records, mode)
    kb_map = {
        "human_partial": parse_human_authored_knowledge(knowledge_dir / "project_knowledge_partial.md"),
        "human_full": parse_human_authored_knowledge(knowledge_dir / "project_knowledge_full.md"),
        "human_noisy": parse_human_authored_knowledge(knowledge_dir / "project_knowledge_noisy.md"),
    }
    md_pair_caches = {name: {} for name in kb_map}
    rows = []
    for threshold in thresholds:
        base, _labels = evaluate_prepared_baseline(prepared, threshold=threshold)
        base["same_project_floor"] = same_project_floor
        rows.append(compact_row(base))
        for condition, kb in kb_map.items():
            row, _labels = evaluate_prepared_md(
                prepared,
                condition=condition,
                kb=kb,
                threshold=threshold,
                same_project_floor=same_project_floor,
                policy=MdPolicy(),
                md_pairs_cache=md_pair_caches[condition],
            )
            rows.append(compact_row(row))
    return rows


def run_floor_sweep(
    records: list[dict[str, Any]],
    *,
    knowledge_dir: Path,
    mode: str,
    threshold: float,
    floors: list[float],
) -> list[dict[str, Any]]:
    prepared = prepare_experiment(records, mode)
    kb_map = {
        "human_partial": parse_human_authored_knowledge(knowledge_dir / "project_knowledge_partial.md"),
        "human_full": parse_human_authored_knowledge(knowledge_dir / "project_knowledge_full.md"),
        "human_noisy": parse_human_authored_knowledge(knowledge_dir / "project_knowledge_noisy.md"),
    }
    md_pair_caches = {name: {} for name in kb_map}
    rows = []
    for floor in floors:
        for condition, kb in kb_map.items():
            row, _labels = evaluate_prepared_md(
                prepared,
                condition=condition,
                kb=kb,
                threshold=threshold,
                same_project_floor=floor,
                policy=MdPolicy(),
                md_pairs_cache=md_pair_caches[condition],
            )
            rows.append(compact_row(row))
    return rows


def run_ablation(
    records: list[dict[str, Any]],
    *,
    knowledge_dir: Path,
    mode: str,
    threshold: float,
    same_project_floor: float,
) -> list[dict[str, Any]]:
    prepared = prepare_experiment(records, mode)
    kb = parse_human_authored_knowledge(knowledge_dir / "project_knowledge_full.md")
    rows = []
    base, _labels = evaluate_prepared_baseline(prepared, threshold=threshold)
    base["same_project_floor"] = same_project_floor
    base["ablation"] = "baseline"
    rows.append(compact_row(base))

    variants = [
        ("human_full_default", "none", MdPolicy()),
        ("human_full_no_cannot_merge", "remove_cannot_merge", MdPolicy(use_cannot_merge=False)),
        ("human_full_no_same_project_floor", "remove_same_project_floor", MdPolicy(use_same_project_floor=False)),
        ("human_full_no_weak_boost", "remove_weak_boost", MdPolicy(use_weak_boost=False)),
        ("human_full_no_generic_safety", "remove_generic_mention_safety", MdPolicy(generic_safety=False)),
    ]
    caches: dict[str, tuple[dict[tuple[int, int], MdPairInfo], Counter]] = {}
    for condition, ablation, policy in variants:
        row, _labels = evaluate_prepared_md(
            prepared,
            condition=condition,
            kb=kb,
            threshold=threshold,
            same_project_floor=same_project_floor,
            policy=policy,
            md_pairs_cache=caches,
        )
        row["ablation"] = ablation
        rows.append(compact_row(row))
    return rows


def aggregate_rows(rows: list[dict[str, Any]], group_key: str = "condition") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(row)

    out = []
    for group, group_rows in grouped.items():
        summary = {group_key: group, "n": len(group_rows)}
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in group_rows if row.get(key) not in (None, "")]
            if not values:
                continue
            summary[f"{key}_mean"] = round(statistics.mean(values), 6)
            summary[f"{key}_std"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
            summary[f"{key}_min"] = round(min(values), 6)
            summary[f"{key}_max"] = round(max(values), 6)
        out.append(summary)
    return sorted(out, key=lambda row: row[group_key])


def run_multi_subset(
    source_mentions: Path,
    *,
    subset_size: int,
    seeds: list[int],
    out_dir: Path,
    knowledge_dir: Path,
    mode: str,
    threshold: float,
    same_project_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = read_jsonl(source_mentions)
    all_rows = []
    subset_summaries = []
    subset_dir = out_dir / "multi_subset_samples"
    for seed in seeds:
        selected_native, diagnostics = choose_subset(source_rows, subset_size, seed)
        compatible = [to_compatible_record(row, index) for index, row in enumerate(selected_native, start=1)]
        seed_dir = subset_dir / f"seed_{seed}"
        write_jsonl(seed_dir / "mentions_external_test_120_native.jsonl", selected_native)
        write_jsonl(seed_dir / "mentions_external_test_120.jsonl", compatible)
        summary = {
            "seed": seed,
            "sampling": diagnostics,
            "native_subset_summary": summarize_subset(selected_native),
            "compatible_subset_summary": summarize_subset(compatible),
        }
        write_json(seed_dir / "external_test_120_summary.json", summary)
        subset_summaries.append({
            "seed": seed,
            **{f"subset_{key}": value for key, value in summary["compatible_subset_summary"].items() if not isinstance(value, dict)},
        })

        result_rows, _labels = evaluate_main_conditions(
            compatible,
            knowledge_dir=knowledge_dir,
            mode=mode,
            threshold=threshold,
            same_project_floor=same_project_floor,
        )
        for row in result_rows:
            compact = compact_row(row)
            compact["seed"] = seed
            all_rows.append(compact)
    return all_rows, subset_summaries


def run_classic_baselines(
    records: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    thresholds = thresholds or {
        "exact": 1.0,
        "jaccard": 0.50,
        "lexical": 0.55,
        "proposed_signature": 0.62,
    }
    modes = list(thresholds)
    items = _prepare_items(_records_to_items(records), modes)
    gold_labels = [item["gold_cluster_id"] for item in items]
    rows = []
    for mode, threshold in thresholds.items():
        labels = cluster_items(items, mode, threshold)
        metrics = evaluate_cluster_labels(gold_labels, labels)
        errors = pair_error_sets_for_labels(gold_labels, labels)
        row = {
            "condition": mode,
            "role": "classic_baseline",
            "threshold": threshold,
            "record_count": len(items),
            "gold_cluster_count": len(set(gold_labels)),
            "pred_cluster_count": len(set(labels)),
            **{key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()},
            "false_merge_by_case_type": case_pair_counter(items, errors["false_merges"]),
            "false_split_by_case_type": case_pair_counter(items, errors["false_splits"]),
        }
        rows.append(compact_row(row))
    return rows


def pair_error_sets_for_labels(gold_labels: list[str], pred_labels: list[str]) -> dict[str, set[tuple[int, int]]]:
    false_merges = set()
    false_splits = set()
    for i in range(len(gold_labels)):
        for j in range(i + 1, len(gold_labels)):
            gold_same = gold_labels[i] == gold_labels[j]
            pred_same = pred_labels[i] == pred_labels[j]
            if pred_same and not gold_same:
                false_merges.add((i, j))
            elif gold_same and not pred_same:
                false_splits.add((i, j))
    return {"false_merges": false_merges, "false_splits": false_splits}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("distribution")]


def parse_upper(value: str) -> float:
    value = str(value).strip().lower()
    if value in {"inf", "+inf", "infinity"}:
        return math.inf
    return float(value)


def bin_values(values: list[float], bin_rows: list[dict[str, str]]) -> list[int]:
    uppers = [parse_upper(row["upper"]) for row in bin_rows]
    counts = [0 for _ in bin_rows]
    for value in values:
        for idx, upper in enumerate(uppers):
            if value <= upper:
                counts[idx] += 1
                break
        else:
            counts[-1] += 1
    return counts


def ratios(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total <= 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def kl_divergence(p: list[float], q: list[float], eps: float = 1e-9) -> float:
    return sum((pi + eps) * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def js_divergence(p: list[float], q: list[float]) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def total_variation(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def emd_bin_index(p: list[float], q: list[float]) -> float:
    if not p:
        return 0.0
    acc = 0.0
    total = 0.0
    for pi, qi in zip(p, q):
        acc += pi - qi
        total += abs(acc)
    return total / max(1, len(p) - 1)


def parse_email_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value[:16], "%Y-%m-%d %H:%M")
    except Exception:
        return None


def synthetic_distribution_values(emails: list[dict[str, Any]]) -> dict[str, list[float]]:
    body_lengths = [len(str(row.get("body") or "")) for row in emails]
    subject_lengths = [len(str(row.get("subject") or "")) for row in emails]
    to_counts = [float(row.get("to_count") or 0) for row in emails]
    cc_counts = [float(row.get("cc_count") or 0) for row in emails]
    attachment_counts = [float(row.get("attachment_count") or 0) for row in emails]
    recipient_counts = [to + cc for to, cc in zip(to_counts, cc_counts)]

    thread_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in emails:
        thread_groups[str(row.get("thread_id") or row.get("email_id") or "")].append(row)
    mails_per_thread = [len(rows) for rows in thread_groups.values()]
    thread_text_chars = [
        sum(len(str(row.get("subject") or "")) + len(str(row.get("body") or "")) for row in rows)
        for rows in thread_groups.values()
    ]

    times = sorted([dt for dt in (parse_email_time(str(row.get("time") or "")) for row in emails) if dt])
    interarrival = [
        max(0.0, (right - left).total_seconds() / 60.0)
        for left, right in zip(times, times[1:])
    ]

    return {
        "body_chars_full_clean": [float(v) for v in body_lengths],
        "body_chars_after_mailtodo_limit": [float(min(v, 1500)) for v in body_lengths],
        "subject_chars": [float(v) for v in subject_lengths],
        "to_count": to_counts,
        "cc_count": cc_counts,
        "recipient_total_count": recipient_counts,
        "attachment_count": attachment_counts,
        "mails_per_thread": [float(v) for v in mails_per_thread],
        "thread_text_chars": [float(v) for v in thread_text_chars],
        "interarrival_minutes": interarrival,
    }


def run_distribution_similarity(
    *,
    profile_bins: Path,
    synthetic_emails: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real_rows = read_csv_dicts(profile_bins)
    real_by_dist: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in real_rows:
        real_by_dist[row["distribution"]].append(row)

    emails = read_jsonl(synthetic_emails)
    values_by_dist = synthetic_distribution_values(emails)

    summary_rows = []
    synthetic_bin_rows = []
    for dist_name, bin_rows in sorted(real_by_dist.items()):
        if dist_name not in values_by_dist:
            continue
        real_counts = [int(float(row.get("count") or 0)) for row in bin_rows]
        real_ratios = ratios(real_counts)
        synth_counts = bin_values(values_by_dist[dist_name], bin_rows)
        synth_ratios = ratios(synth_counts)
        summary_rows.append({
            "distribution": dist_name,
            "real_n": sum(real_counts),
            "synthetic_n": sum(synth_counts),
            "kl_real_to_synthetic": round(kl_divergence(real_ratios, synth_ratios), 6),
            "js_divergence": round(js_divergence(real_ratios, synth_ratios), 6),
            "total_variation": round(total_variation(real_ratios, synth_ratios), 6),
            "emd_bin_index": round(emd_bin_index(real_ratios, synth_ratios), 6),
        })
        for row, count, ratio in zip(bin_rows, synth_counts, synth_ratios):
            synthetic_bin_rows.append({
                "distribution": dist_name,
                "bin": row.get("bin"),
                "upper": row.get("upper"),
                "synthetic_count": count,
                "synthetic_ratio": round(ratio, 6),
                "real_count": row.get("count"),
                "real_ratio": row.get("ratio"),
            })
    return summary_rows, synthetic_bin_rows


def create_markdown_summary(
    path: Path,
    *,
    payload: dict[str, Any],
    main_rows: list[dict[str, Any]],
    multi_summary: list[dict[str, Any]],
) -> None:
    best = max([row for row in main_rows if row["condition"] != "baseline_signature"], key=lambda row: row["bcubed_f1"])
    baseline = next(row for row in main_rows if row["condition"] == "baseline_signature")
    lines = [
        "# IEEE Access Experiment Summary",
        "",
        f"- Created at: {payload['created_at']}",
        f"- Dataset: `{payload['dataset']}`",
        f"- Source mentions: `{payload['source_mentions']}`",
        f"- Threshold: {payload['threshold']}",
        f"- Same-project floor: {payload['same_project_floor']}",
        "",
        "## Main Result",
        "",
        "| Condition | B-cubed F1 | Pairwise F1 | False Merge | False Split | Pred Clusters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['condition']} | {row['bcubed_f1']:.6f} | {row['pairwise_f1']:.6f} | "
            f"{row['false_merge_count']} | {row['false_split_count']} | {row['pred_cluster_count']} |"
        )
    lines.extend([
        "",
        "## Recommended IEEE Access Claim",
        "",
        (
            f"The best MD condition is `{best['condition']}`, improving B-cubed F1 from "
            f"{baseline['bcubed_f1']:.6f} to {best['bcubed_f1']:.6f} while reducing false merge "
            f"from {baseline['false_merge_count']} to {best['false_merge_count']}."
        ),
        "",
        "## Multi-subset Robustness",
        "",
        "| Condition | B-cubed F1 mean | B-cubed F1 std | False merge mean | False split mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in multi_summary:
        lines.append(
            f"| {row['condition']} | {row.get('bcubed_f1_mean', 0):.6f} | {row.get('bcubed_f1_std', 0):.6f} | "
            f"{row.get('false_merge_count_mean', 0):.3f} | {row.get('false_split_count_mean', 0):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IEEE Access robustness experiments.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Main 120-record JSONL dataset.")
    parser.add_argument("--source-mentions", default=DEFAULT_SOURCE_MENTIONS, help="398-record matched mentions JSONL.")
    parser.add_argument("--knowledge-dir", default=DEFAULT_KNOWLEDGE_DIR, help="Knowledge MD directory.")
    parser.add_argument("--synthetic-emails", default=DEFAULT_SYNTHETIC_EMAILS, help="Synthetic raw emails JSONL.")
    parser.add_argument("--profile-bins", default=DEFAULT_PROFILE_BINS, help="Real profile distribution bins CSV.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory. Supports {timestamp}.")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="Baseline Project Signature mode.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Default clustering threshold.")
    parser.add_argument("--same-project-floor", type=float, default=0.96, help="Default MD same-project floor.")
    parser.add_argument("--thresholds", default=",".join(f"{x:.2f}" for x in DEFAULT_THRESHOLDS), help="Comma-separated threshold sweep values.")
    parser.add_argument("--floors", default=",".join(f"{x:.2f}" for x in DEFAULT_FLOORS), help="Comma-separated floor sweep values.")
    parser.add_argument("--subset-seeds", default=",".join(str(x) for x in DEFAULT_SUBSET_SEEDS), help="Comma-separated multi-subset seeds.")
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE, help="Balanced subset size.")
    parser.add_argument("--skip-threshold-sweep", action="store_true")
    parser.add_argument("--skip-floor-sweep", action="store_true")
    parser.add_argument("--skip-multi-subset", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--skip-classic-baselines", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir.replace("{timestamp}", timestamp)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Path(args.dataset)
    source_mentions = Path(args.source_mentions)
    knowledge_dir = Path(args.knowledge_dir)
    synthetic_emails = Path(args.synthetic_emails)
    profile_bins = Path(args.profile_bins)
    thresholds = [float(x) for x in parse_number_list(args.thresholds)]
    floors = [float(x) for x in parse_number_list(args.floors)]
    subset_seeds = [int(x) for x in parse_number_list(args.subset_seeds, as_int=True)]

    print(f"[load] main dataset={dataset}")
    records = read_jsonl(dataset)

    print("[main] evaluating baseline and MD variants")
    main_rows, labels_by_condition = evaluate_main_conditions(
        records,
        knowledge_dir=knowledge_dir,
        mode=args.mode,
        threshold=args.threshold,
        same_project_floor=args.same_project_floor,
    )
    main_compact = [compact_row(row) for row in main_rows]
    write_csv_rows(out_dir / "main_comparison.csv", main_compact)
    write_labels_jsonl(out_dir / "main_comparison_labels.jsonl", _records_to_items(records), labels_by_condition)

    threshold_rows = []
    if not args.skip_threshold_sweep:
        print("[sweep] threshold sensitivity")
        threshold_rows = run_threshold_sweep(
            records,
            knowledge_dir=knowledge_dir,
            mode=args.mode,
            thresholds=thresholds,
            same_project_floor=args.same_project_floor,
        )
        write_csv_rows(out_dir / "threshold_sweep.csv", threshold_rows)

    floor_rows = []
    if not args.skip_floor_sweep:
        print("[sweep] same-project floor sensitivity")
        floor_rows = run_floor_sweep(
            records,
            knowledge_dir=knowledge_dir,
            mode=args.mode,
            threshold=args.threshold,
            floors=floors,
        )
        write_csv_rows(out_dir / "floor_sweep.csv", floor_rows)

    ablation_rows = []
    if not args.skip_ablation:
        print("[ablation] MD component ablation")
        ablation_rows = run_ablation(
            records,
            knowledge_dir=knowledge_dir,
            mode=args.mode,
            threshold=args.threshold,
            same_project_floor=args.same_project_floor,
        )
        write_csv_rows(out_dir / "ablation_human_full.csv", ablation_rows)

    multi_rows = []
    multi_subset_summaries = []
    multi_summary = []
    if not args.skip_multi_subset:
        print("[robustness] multi-subset evaluation")
        multi_rows, multi_subset_summaries = run_multi_subset(
            source_mentions,
            subset_size=args.subset_size,
            seeds=subset_seeds,
            out_dir=out_dir,
            knowledge_dir=knowledge_dir,
            mode=args.mode,
            threshold=args.threshold,
            same_project_floor=args.same_project_floor,
        )
        multi_summary = aggregate_rows(multi_rows)
        write_csv_rows(out_dir / "multi_subset_results.csv", multi_rows)
        write_csv_rows(out_dir / "multi_subset_summary.csv", multi_summary)
        write_csv_rows(out_dir / "multi_subset_sample_summaries.csv", multi_subset_summaries)

    distribution_rows = []
    synthetic_bin_rows = []
    if not args.skip_distribution:
        print("[distribution] synthetic vs profile similarity")
        distribution_rows, synthetic_bin_rows = run_distribution_similarity(
            profile_bins=profile_bins,
            synthetic_emails=synthetic_emails,
        )
        write_csv_rows(out_dir / "distribution_similarity.csv", distribution_rows)
        write_csv_rows(out_dir / "synthetic_distribution_bins.csv", synthetic_bin_rows)

    classic_rows = []
    if not args.skip_classic_baselines:
        print("[baseline] classic baseline sanity checks")
        classic_rows = run_classic_baselines(records)
        write_csv_rows(out_dir / "classic_baselines.csv", classic_rows)

    deltas = [comparison_delta(main_rows[0], row) for row in main_rows[1:]]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "source_mentions": str(source_mentions),
        "knowledge_dir": str(knowledge_dir),
        "synthetic_emails": str(synthetic_emails),
        "profile_bins": str(profile_bins),
        "mode": args.mode,
        "threshold": args.threshold,
        "same_project_floor": args.same_project_floor,
        "thresholds": thresholds,
        "floors": floors,
        "subset_seeds": subset_seeds,
        "main_results": main_rows,
        "deltas_vs_baseline": deltas,
        "threshold_sweep_count": len(threshold_rows),
        "floor_sweep_count": len(floor_rows),
        "ablation_rows": ablation_rows,
        "multi_subset_summary": multi_summary,
        "distribution_similarity": distribution_rows,
        "classic_baselines": classic_rows,
        "outputs": {
            "main_comparison_csv": str(out_dir / "main_comparison.csv"),
            "threshold_sweep_csv": str(out_dir / "threshold_sweep.csv"),
            "floor_sweep_csv": str(out_dir / "floor_sweep.csv"),
            "ablation_csv": str(out_dir / "ablation_human_full.csv"),
            "multi_subset_results_csv": str(out_dir / "multi_subset_results.csv"),
            "multi_subset_summary_csv": str(out_dir / "multi_subset_summary.csv"),
            "distribution_similarity_csv": str(out_dir / "distribution_similarity.csv"),
            "classic_baselines_csv": str(out_dir / "classic_baselines.csv"),
        },
    }
    write_json(out_dir / "ieee_access_experiment_summary.json", payload)
    create_markdown_summary(
        out_dir / "IEEE_ACCESS_EXPERIMENT_SUMMARY.md",
        payload=payload,
        main_rows=main_rows,
        multi_summary=multi_summary,
    )

    compact = {
        "out_dir": str(out_dir),
        "main": [
            {
                "condition": row["condition"],
                "bcubed_f1": row["bcubed_f1"],
                "pairwise_f1": row["pairwise_f1"],
                "false_merge_count": row["false_merge_count"],
                "false_split_count": row["false_split_count"],
            }
            for row in main_rows
        ],
        "multi_subset_summary": multi_summary,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
