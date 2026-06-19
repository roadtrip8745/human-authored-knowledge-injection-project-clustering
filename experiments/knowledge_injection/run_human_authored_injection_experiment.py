# -*- coding: utf-8 -*-
"""
Compare baseline Project Signature clustering with human-authored knowledge injection.

This script intentionally keeps the previous JIPS baseline untouched. It imports
the existing evaluator/scoring code from the prior project and runs human-authored knowledge
as a thin pairwise-score adjustment layer:

- same MD project/canonical match: raise the pair score to a high confidence
- MD cannot-merge relation: block the pair merge
- otherwise: use the original Project Signature score unchanged

HITL feedback files are not used in this experiment; they are left for future
work or optional extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PREVIOUS_RESEARCH_ROOT = os.environ.get(
    "PROJECT_SIGNATURE_BASELINE_ROOT",
    str(Path(__file__).resolve().parents[2] / "external" / "mail-to-report-research"),
)
DEFAULT_DATASET = r"data\synthetic\extraction_run_20260618_085129\external_test_120\mentions_external_test_120.jsonl"
DEFAULT_KNOWLEDGE_DIR = r"data\synthetic\human_authored_knowledge"
DEFAULT_OUT_DIR = r"data\synthetic\human_authored_injection_runs\run_{timestamp}"
DEFAULT_MODE = "proposed_signature"
DEFAULT_THRESHOLD = 0.62


def ensure_previous_code_importable() -> None:
    root = str(Path(PREVIOUS_RESEARCH_ROOT))
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_previous_code_importable()

from backend.services.project_normalization_svc import lexical_score, normalize_for_compare  # noqa: E402
from backend.services.research_eval_svc import (  # noqa: E402
    _UnionFind,
    _cluster_merge_is_consistent,
    _has_conflict,
    _prepare_items,
    _records_to_items,
    _score_pair,
    cluster_items,
)
from backend.services.research_metrics_svc import evaluate_cluster_labels  # noqa: E402


@dataclass
class KnowledgeProject:
    canonical: str
    confidence: str = ""
    status: str = ""
    system: str = ""
    modules: list[str] = field(default_factory=list)
    work_types: list[str] = field(default_factory=list)
    owner_group: str = ""
    aliases: list[str] = field(default_factory=list)
    cannot_merge_with: list[str] = field(default_factory=list)


@dataclass
class KnowledgeBase:
    path: str
    projects: dict[str, KnowledgeProject]
    alias_to_canonical: dict[str, str]
    cannot_pairs: set[tuple[str, str]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no} JSON parse failed: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm(value: str) -> str:
    return normalize_for_compare(value or "")


def is_generic_short_mention(value: str) -> bool:
    key = norm(value)
    if not key:
        return False
    if key in {"erp", "mes", "wms", "scm", "oms", "qms", "itsm", "dw", "pi"}:
        return True
    return len(key) <= 3 and re.fullmatch(r"[a-z0-9]+", key) is not None


def is_specific_alias(value: str) -> bool:
    key = norm(value)
    if not key:
        return False
    return len(key) >= 5 or " " in key


def split_csv_value(value: str) -> list[str]:
    value = (value or "").strip()
    if not value or value == "없음" or value == "미상":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_human_authored_knowledge(path: Path) -> KnowledgeBase:
    text = path.read_text(encoding="utf-8-sig")
    projects: dict[str, KnowledgeProject] = {}

    current: KnowledgeProject | None = None
    active_list: str | None = None

    def flush() -> None:
        if current and current.canonical:
            projects[norm(current.canonical)] = current

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            title = stripped[3:].strip()
            # Ignore non-project note sections in noisy knowledge.
            if title in {"미확정 약어 메모", "폐기된 프로젝트명 메모", "Alias Corrections", "Cannot Merge Corrections", "Review Notes"}:
                current = None
            else:
                current = KnowledgeProject(canonical=title)
            active_list = None
            continue

        if current is None:
            continue

        if stripped.startswith("- "):
            body = stripped[2:].strip()
            if body.endswith(":"):
                active_list = body[:-1].strip()
                continue
            if ":" in body and active_list is None:
                key, value = [part.strip() for part in body.split(":", 1)]
                if key == "confidence":
                    current.confidence = value
                elif key == "status":
                    current.status = value
                elif key == "system":
                    current.system = value
                elif key == "modules":
                    current.modules = split_csv_value(value)
                elif key == "work_types":
                    current.work_types = split_csv_value(value)
                elif key == "owner_group":
                    current.owner_group = value
                continue

        if stripped.startswith("- aliases:"):
            active_list = "aliases"
            continue
        if stripped.startswith("- cannot_merge_with:"):
            active_list = "cannot_merge_with"
            continue
        if stripped.startswith("- notes:"):
            active_list = "notes"
            continue

        if stripped.startswith("- ") and active_list:
            item = stripped[2:].strip()
            if not item or item == "없음":
                continue
            if active_list == "aliases":
                current.aliases.append(item)
            elif active_list == "cannot_merge_with":
                current.cannot_merge_with.append(item)

    flush()

    alias_to_canonical: dict[str, str] = {}
    alias_collisions = set()
    for project in projects.values():
        for alias in [project.canonical, *project.aliases]:
            key = norm(alias)
            if not key:
                continue
            if key in alias_to_canonical and alias_to_canonical[key] != project.canonical:
                alias_collisions.add(key)
                continue
            alias_to_canonical[key] = project.canonical
    for key in alias_collisions:
        alias_to_canonical.pop(key, None)

    cannot_pairs = set()
    for project in projects.values():
        left = project.canonical
        for right in project.cannot_merge_with:
            if norm(right) not in projects:
                continue
            cannot_pairs.add(canonical_pair(left, right))

    return KnowledgeBase(
        path=str(path),
        projects=projects,
        alias_to_canonical=alias_to_canonical,
        cannot_pairs=cannot_pairs,
    )


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    a, b = sorted([norm(left), norm(right)])
    return a, b


def item_text(item: dict[str, Any]) -> str:
    meta = item.get("metadata") or {}
    return "\n".join(
        [
            str(item.get("mention") or ""),
            str(meta.get("mail_subject") or ""),
            str(item.get("context") or "")[:900],
        ]
    )


def match_item_to_kb(item: dict[str, Any], kb: KnowledgeBase) -> tuple[str, float, str]:
    mention = str(item.get("mention") or "")
    mention_key = norm(mention)
    generic_mention = is_generic_short_mention(mention)
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
            elif is_specific_alias(alias) and alias_key in mention_key:
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


def md_adjusted_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    kb: KnowledgeBase,
    mode: str,
    same_project_floor: float,
) -> tuple[float, bool, dict[str, Any]]:
    base_score = _score_pair(a, b, mode)
    base_conflict = _has_conflict(a, b, mode)

    a_project, a_conf, a_evidence = match_item_to_kb(a, kb)
    b_project, b_conf, b_evidence = match_item_to_kb(b, kb)

    md_conflict = False
    adjusted = base_score
    reason = "baseline"

    if a_project and b_project:
        if canonical_pair(a_project, b_project) in kb.cannot_pairs:
            md_conflict = True
            adjusted = 0.0
            reason = "md_cannot_merge"
        elif norm(a_project) == norm(b_project):
            min_conf = min(a_conf, b_conf)
            if min_conf >= 0.90:
                adjusted = max(base_score, same_project_floor)
                reason = "md_same_project"
            elif min_conf >= 0.82:
                # Context-only MD matches are useful but too weak to force a
                # merge. Keep the boost below the hard confidence floor.
                adjusted = max(base_score, min(0.80, base_score + 0.08))
                reason = "md_weak_same_project"

    return max(0.0, min(1.0, adjusted)), bool(base_conflict or md_conflict), {
        "reason": reason,
        "base_score": round(base_score, 4),
        "adjusted_score": round(adjusted, 4),
        "a_project": a_project,
        "b_project": b_project,
        "a_match_confidence": round(a_conf, 4),
        "b_match_confidence": round(b_conf, 4),
        "a_match_evidence": a_evidence,
        "b_match_evidence": b_evidence,
    }


def cluster_items_with_md(
    items: list[dict[str, Any]],
    *,
    mode: str,
    threshold: float,
    kb: KnowledgeBase,
    same_project_floor: float,
) -> tuple[list[str], dict[str, Any]]:
    n = len(items)
    uf = _UnionFind(n)
    pair_scores = {}
    pair_conflicts = {}
    pair_debug = {}
    adjustment_counts = Counter()

    for i in range(n):
        for j in range(i + 1, n):
            score, conflict, debug = md_adjusted_pair(
                items[i],
                items[j],
                kb=kb,
                mode=mode,
                same_project_floor=same_project_floor,
            )
            pair_scores[(i, j)] = score
            pair_conflicts[(i, j)] = conflict
            pair_debug[(i, j)] = debug
            adjustment_counts[debug["reason"]] += 1

    ranked = sorted(pair_scores.items(), key=lambda kv: -kv[1])
    review_threshold = max(0.0, threshold - 0.12)

    blocked_by_md = 0
    merged_by_md = 0
    total_unions = 0

    for (i, j), score in ranked:
        if score < threshold:
            break
        if pair_conflicts[(i, j)]:
            if pair_debug[(i, j)]["reason"] == "md_cannot_merge":
                blocked_by_md += 1
            continue
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if not _cluster_merge_is_consistent(uf, ri, rj, pair_scores, pair_conflicts, review_threshold):
            continue
        if pair_debug[(i, j)]["reason"] == "md_same_project":
            merged_by_md += 1
        total_unions += 1
        uf.union(ri, rj)

    root_to_label = {}
    labels = []
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_label:
            root_to_label[root] = f"MD{len(root_to_label) + 1:04d}"
        labels.append(root_to_label[root])

    debug_summary = {
        "knowledge_file": kb.path,
        "knowledge_project_count": len(kb.projects),
        "knowledge_alias_count": len(kb.alias_to_canonical),
        "knowledge_cannot_pair_count": len(kb.cannot_pairs),
        "pair_adjustment_counts": dict(adjustment_counts),
        "blocked_by_md": blocked_by_md,
        "merged_by_md": merged_by_md,
        "total_unions": total_unions,
    }
    return labels, debug_summary


def round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def pair_error_sets(gold_labels: list[str], pred_labels: list[str]) -> dict[str, set[tuple[int, int]]]:
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


def pair_examples(items: list[dict[str, Any]], pairs: set[tuple[int, int]], limit: int = 8) -> list[dict[str, Any]]:
    examples = []
    for i, j in sorted(pairs)[:limit]:
        a = items[i]
        b = items[j]
        examples.append(
            {
                "a_record_id": a.get("record_id"),
                "b_record_id": b.get("record_id"),
                "a_mention": a.get("mention"),
                "b_mention": b.get("mention"),
                "a_gold": a.get("gold_cluster_id"),
                "b_gold": b.get("gold_cluster_id"),
                "a_case_type": (a.get("metadata") or {}).get("case_type"),
                "b_case_type": (b.get("metadata") or {}).get("case_type"),
            }
        )
    return examples


def case_pair_counter(items: list[dict[str, Any]], pairs: set[tuple[int, int]]) -> dict[str, int]:
    counter = Counter()
    for i, j in pairs:
        ca = (items[i].get("metadata") or {}).get("case_type") or "UNKNOWN"
        cb = (items[j].get("metadata") or {}).get("case_type") or "UNKNOWN"
        counter[ca if ca == cb else f"{ca}+{cb}"] += 1
    return dict(counter.most_common())


def condition_result(
    *,
    name: str,
    role: str,
    items: list[dict[str, Any]],
    gold_labels: list[str],
    labels: list[str],
    threshold: float,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = round_metrics(evaluate_cluster_labels(gold_labels, labels))
    errors = pair_error_sets(gold_labels, labels)
    result = {
        "condition": name,
        "role": role,
        "threshold": threshold,
        "record_count": len(items),
        "gold_cluster_count": len(set(gold_labels)),
        "pred_cluster_count": len(set(labels)),
        **metrics,
        "false_merge_by_case_type": case_pair_counter(items, errors["false_merges"]),
        "false_split_by_case_type": case_pair_counter(items, errors["false_splits"]),
        "false_merge_examples": pair_examples(items, errors["false_merges"]),
        "false_split_examples": pair_examples(items, errors["false_splits"]),
    }
    if debug:
        result["md_debug"] = debug
    return result


def comparison_delta(baseline: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    def delta(key: str) -> float:
        return round(float(condition.get(key, 0)) - float(baseline.get(key, 0)), 6)

    base_fm = int(baseline.get("false_merge_count", 0))
    cond_fm = int(condition.get("false_merge_count", 0))
    base_fs = int(baseline.get("false_split_count", 0))
    cond_fs = int(condition.get("false_split_count", 0))
    return {
        "condition": condition["condition"],
        "pairwise_f1_delta": delta("pairwise_f1"),
        "bcubed_f1_delta": delta("bcubed_f1"),
        "false_merge_delta": cond_fm - base_fm,
        "false_split_delta": cond_fs - base_fs,
        "false_merge_reduction_pct": round(((base_fm - cond_fm) / base_fm * 100.0), 2) if base_fm else 0.0,
        "false_split_reduction_pct": round(((base_fs - cond_fs) / base_fs * 100.0), 2) if base_fs else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "condition",
        "role",
        "threshold",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_labels_jsonl(path: Path, items: list[dict[str, Any]], labels_by_condition: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for idx, item in enumerate(items):
            row = {
                "record_id": item.get("record_id"),
                "mention": item.get("mention"),
                "gold_cluster_id": item.get("gold_cluster_id"),
                "case_type": (item.get("metadata") or {}).get("case_type"),
                "predicted": {name: labels[idx] for name, labels in labels_by_condition.items()},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit_progress(progress, step: str, detail: str = "", current: int | None = None, total: int | None = None) -> None:
    if progress is None:
        return
    progress({
        "step": step,
        "detail": detail,
        "current": current,
        "total": total,
        "time": datetime.now().isoformat(timespec="seconds"),
    })


def run_experiment(
    *,
    dataset: str = DEFAULT_DATASET,
    knowledge_dir: str = DEFAULT_KNOWLEDGE_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    mode: str = DEFAULT_MODE,
    threshold: float = DEFAULT_THRESHOLD,
    same_project_floor: float = 0.96,
    progress=None,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    knowledge_path = Path(knowledge_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(out_dir.replace("{timestamp}", timestamp)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    emit_progress(progress, "데이터 로드", str(dataset_path), 1, 7)
    records = read_jsonl(dataset_path)

    emit_progress(progress, "baseline 입력 준비", f"{len(records)} records", 2, 7)
    items = _records_to_items(records)
    prepared = _prepare_items(items, [mode])
    gold_labels = [item["gold_cluster_id"] for item in prepared]

    emit_progress(progress, "baseline 실행", mode, 3, 7)
    baseline_labels = cluster_items(prepared, mode, threshold)
    labels_by_condition = {"baseline_signature": baseline_labels}
    results = [
        condition_result(
            name="baseline_signature",
            role="baseline",
            items=prepared,
            gold_labels=gold_labels,
            labels=baseline_labels,
            threshold=threshold,
        )
    ]

    knowledge_conditions = [
        ("human_partial", knowledge_path / "project_knowledge_partial.md"),
        ("human_full", knowledge_path / "project_knowledge_full.md"),
        ("human_noisy", knowledge_path / "project_knowledge_noisy.md"),
    ]

    for idx, (name, path) in enumerate(knowledge_conditions, start=4):
        emit_progress(progress, f"{name} 실행", str(path), idx, 7)
        kb = parse_human_authored_knowledge(path)
        labels, debug = cluster_items_with_md(
            prepared,
            mode=mode,
            threshold=threshold,
            kb=kb,
            same_project_floor=same_project_floor,
        )
        labels_by_condition[name] = labels
        results.append(
            condition_result(
                name=name,
                role="human_authored_knowledge",
                items=prepared,
                gold_labels=gold_labels,
                labels=labels,
                threshold=threshold,
                debug=debug,
            )
        )

    baseline = results[0]
    deltas = [comparison_delta(baseline, row) for row in results[1:]]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "knowledge_dir": str(knowledge_path),
        "mode": mode,
        "threshold": threshold,
        "same_project_floor": same_project_floor,
        "results": results,
        "deltas_vs_baseline": deltas,
        "outputs": {
            "json": str(output_dir / "human_authored_injection_comparison.json"),
            "csv": str(output_dir / "human_authored_injection_comparison.csv"),
            "labels": str(output_dir / "human_authored_injection_labels.jsonl"),
        },
    }

    emit_progress(progress, "결과 저장", str(output_dir), 7, 7)
    write_json(output_dir / "human_authored_injection_comparison.json", payload)
    write_csv(output_dir / "human_authored_injection_comparison.csv", results)
    write_labels_jsonl(output_dir / "human_authored_injection_labels.jsonl", prepared, labels_by_condition)
    emit_progress(progress, "완료", str(output_dir), 7, 7)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs human-authored knowledge injection comparison.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to 120-record JSONL subset.")
    parser.add_argument("--knowledge-dir", default=DEFAULT_KNOWLEDGE_DIR, help="Directory containing project_knowledge_*.md.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory. Supports {timestamp}.")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="Existing evaluator mode to use as baseline.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Clustering threshold.")
    parser.add_argument("--same-project-floor", type=float, default=0.96, help="Minimum score for MD same-project pairs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_experiment(
        dataset=args.dataset,
        knowledge_dir=args.knowledge_dir,
        out_dir=args.out_dir,
        mode=args.mode,
        threshold=args.threshold,
        same_project_floor=args.same_project_floor,
    )
    compact_results = [
        {
            "condition": row["condition"],
            "pairwise_f1": row["pairwise_f1"],
            "bcubed_f1": row["bcubed_f1"],
            "false_merge_count": row["false_merge_count"],
            "false_split_count": row["false_split_count"],
            "pred_cluster_count": row["pred_cluster_count"],
        }
        for row in payload["results"]
    ]
    print(json.dumps({
        "out_dir": str(Path(payload["outputs"]["json"]).parent),
        "results": compact_results,
        "deltas_vs_baseline": payload["deltas_vs_baseline"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
