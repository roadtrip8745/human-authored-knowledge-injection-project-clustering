# -*- coding: utf-8 -*-
"""
Create a 120-record external test subset from extracted synthetic mentions.

The previous JIPS experiment used 120 mention-level records in each test split.
This script builds a comparable external test subset from the end-to-end
raw-email -> LLM extraction output while preserving cluster, case-type, and
week diversity as much as possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = r"data\synthetic\extraction_run_20260618_085129\mentions_extracted.jsonl"
DEFAULT_OUT_DIR = r"data\synthetic\extraction_run_20260618_085129\external_test_120"
DEFAULT_SIZE = 120
DEFAULT_SEED = 20260618

DIFFICULTY_BY_CASE = {
    "base": "easy",
    "lexical_var": "easy",
    "abbrev_var": "medium",
    "general_heavy": "medium",
    "conflict_risk": "hard",
    "context_noise": "medium",
    "chain_risk": "hard",
    "cross_dept": "medium",
    "same_system_module": "hard",
    "multi_project_reference": "hard",
}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_float(seed: int, *parts: Any) -> float:
    text = "|".join([str(seed), *[str(p) for p in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def iso_week(value: str) -> str:
    try:
        dt = datetime.strptime((value or "")[:16], "%Y-%m-%d %H:%M")
        cal = dt.isocalendar()
        return f"{cal.year}-W{cal.week:02d}"
    except Exception:
        return "UNKNOWN"


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata")
    return meta if isinstance(meta, dict) else {}


def case_type(row: dict[str, Any]) -> str:
    return str(row.get("case_type") or metadata(row).get("case_type") or "UNKNOWN")


def source_week(row: dict[str, Any]) -> str:
    meta = metadata(row)
    return str(meta.get("time_window") or iso_week(str(meta.get("time") or "")) or "UNKNOWN")


def inferred_difficulty(row: dict[str, Any]) -> str:
    meta = metadata(row)
    ctype = case_type(row)
    return str(row.get("difficulty") or meta.get("difficulty") or DIFFICULTY_BY_CASE.get(ctype, "medium"))


def inferred_is_cross_dept(row: dict[str, Any]) -> bool:
    meta = metadata(row)
    if "is_cross_dept" in row:
        return bool(row.get("is_cross_dept"))
    if "is_cross_dept" in meta:
        return bool(meta.get("is_cross_dept"))
    return case_type(row) == "cross_dept"


def largest_remainder_quota(
    counts: Counter[str],
    total: int,
    *,
    minimum_for_present: int = 0,
) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key in counts}
    keys = list(counts.keys())
    if not keys:
        return {}

    quotas = {key: 0 for key in keys}
    remaining = total

    if minimum_for_present > 0:
        for key in keys:
            min_value = min(minimum_for_present, counts[key])
            quotas[key] = min_value
            remaining -= min_value

    if remaining < 0:
        # Fall back to one-per-largest group if the requested minimum is too high.
        quotas = {key: 0 for key in keys}
        remaining = total
        for key, _count in counts.most_common(total):
            quotas[key] = 1
            remaining -= 1

    base_total = sum(counts.values())
    if remaining <= 0 or base_total <= 0:
        return quotas

    raw = {}
    for key in keys:
        available = counts[key] - quotas[key]
        share = counts[key] / base_total * remaining
        add = min(available, int(math.floor(share)))
        quotas[key] += add
        raw[key] = share - add

    remaining = total - sum(quotas.values())
    while remaining > 0:
        candidates = [key for key in keys if quotas[key] < counts[key]]
        if not candidates:
            break
        candidates.sort(key=lambda key: (raw.get(key, 0.0), counts[key]), reverse=True)
        quotas[candidates[0]] += 1
        raw[candidates[0]] = 0.0
        remaining -= 1

    return quotas


def normalize_to_size(quotas: dict[str, int], counts: Counter[str], total: int) -> dict[str, int]:
    quotas = {key: min(value, counts[key]) for key, value in quotas.items()}
    while sum(quotas.values()) < total:
        candidates = [key for key in counts if quotas.get(key, 0) < counts[key]]
        if not candidates:
            break
        candidates.sort(key=lambda key: counts[key] - quotas.get(key, 0), reverse=True)
        quotas[candidates[0]] = quotas.get(candidates[0], 0) + 1
    while sum(quotas.values()) > total:
        candidates = [key for key, value in quotas.items() if value > 0]
        candidates.sort(key=lambda key: quotas[key], reverse=True)
        quotas[candidates[0]] -= 1
    return quotas


def choose_subset(rows: list[dict[str, Any]], size: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [row for row in rows if row.get("gold_cluster_id") and row.get("gold_cluster_id") != "UNKNOWN"]
    if len(candidates) < size:
        raise ValueError(f"Not enough matched mentions: requested {size}, available {len(candidates)}")

    cluster_counts = Counter(str(row["gold_cluster_id"]) for row in candidates)
    case_counts = Counter(case_type(row) for row in candidates)
    week_counts = Counter(source_week(row) for row in candidates)

    cluster_quota = largest_remainder_quota(cluster_counts, size, minimum_for_present=2)
    cluster_quota = normalize_to_size(cluster_quota, cluster_counts, size)
    case_quota = largest_remainder_quota(case_counts, size, minimum_for_present=1)
    week_quota = largest_remainder_quota(week_counts, size)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_cluster = Counter()
    selected_case = Counter()
    selected_week = Counter()

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(str(row["gold_cluster_id"]), []).append(row)

    def add_selected(row: dict[str, Any]) -> None:
        selected.append(row)
        selected_ids.add(str(row.get("record_id")))
        selected_cluster[str(row["gold_cluster_id"])] += 1
        selected_case[case_type(row)] += 1
        selected_week[source_week(row)] += 1

    # First reserve quota for rare case types so the 120-record subset remains
    # useful for per-case external analysis.
    for ctype in sorted(case_counts, key=lambda key: (case_counts[key], key)):
        target = min(case_quota.get(ctype, 0), case_counts[ctype])
        while selected_case[ctype] < target:
            remaining = [
                row
                for row in candidates
                if case_type(row) == ctype
                and str(row.get("record_id")) not in selected_ids
                and selected_cluster[str(row["gold_cluster_id"])] < cluster_quota.get(str(row["gold_cluster_id"]), 0)
            ]
            if not remaining:
                break

            def rare_case_score(row: dict[str, Any]) -> tuple[int, int, float]:
                cluster = str(row["gold_cluster_id"])
                week = source_week(row)
                return (
                    cluster_quota.get(cluster, 0) - selected_cluster[cluster],
                    week_quota.get(week, 0) - selected_week[week],
                    -stable_float(seed, row.get("record_id"), row.get("mention")),
                )

            add_selected(max(remaining, key=rare_case_score))

    cluster_order = sorted(
        groups,
        key=lambda cluster: (-cluster_quota.get(cluster, 0), cluster, stable_float(seed, cluster)),
    )

    for cluster in cluster_order:
        target = cluster_quota.get(cluster, 0)
        if target <= 0:
            continue
        pool = list(groups[cluster])
        while selected_cluster[cluster] < target:
            remaining = [row for row in pool if str(row.get("record_id")) not in selected_ids]
            if not remaining:
                break

            def row_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
                ctype = case_type(row)
                week = source_week(row)
                case_need = case_quota.get(ctype, 0) - selected_case[ctype]
                week_need = week_quota.get(week, 0) - selected_week[week]
                # Positive deficits are useful; over-filled strata are allowed but lower priority.
                return (
                    4.0 * case_need + 1.5 * week_need,
                    case_need,
                    week_need,
                    -stable_float(seed, row.get("record_id"), row.get("mention")),
                )

            best = max(remaining, key=row_score)
            add_selected(best)

    if len(selected) != size:
        raise RuntimeError(f"Sampler selected {len(selected)} records, expected {size}")

    diagnostics = {
        "source_count": len(rows),
        "matched_candidate_count": len(candidates),
        "target_size": size,
        "cluster_quota": dict(sorted(cluster_quota.items())),
        "case_quota": dict(sorted(case_quota.items())),
        "week_quota": dict(sorted(week_quota.items())),
    }
    return selected, diagnostics


def to_compatible_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    meta = dict(metadata(row))
    ctype = case_type(row)
    difficulty = str(row.get("difficulty") or meta.get("difficulty") or DIFFICULTY_BY_CASE.get(ctype, "medium"))
    is_cross = bool(row.get("is_cross_dept") if "is_cross_dept" in row else meta.get("is_cross_dept", ctype == "cross_dept"))
    time_window = str(meta.get("time_window") or iso_week(str(meta.get("time") or "")) or "")

    meta.update(
        {
            "case_type": ctype,
            "difficulty": difficulty,
            "seed_version": "OUTLOOK_GPTOSS_EXT120",
            "time_window": time_window,
            "external_subset": "outlook_gptoss_120",
            "original_record_id": row.get("record_id"),
        }
    )

    return {
        "record_id": f"OUTLOOK_EXT_{index:04d}",
        "seed_version": "OUTLOOK_GPTOSS_EXT120",
        "gold_cluster_id": row.get("gold_cluster_id"),
        "mention": row.get("mention"),
        "context": row.get("context"),
        "difficulty": difficulty,
        "split": "test",
        "is_cross_dept": is_cross,
        "metadata": meta,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "cluster_count": len({row.get("gold_cluster_id") for row in rows}),
        "cluster_counts": dict(Counter(str(row.get("gold_cluster_id")) for row in rows).most_common()),
        "case_type_counts": dict(Counter(case_type(row) for row in rows).most_common()),
        "difficulty_counts": dict(Counter(inferred_difficulty(row) for row in rows).most_common()),
        "week_counts": dict(Counter(source_week(row) for row in rows).most_common()),
        "cross_dept_count": sum(1 for row in rows if inferred_is_cross_dept(row)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 120-record external test subset.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to mentions_extracted.jsonl.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Subset size.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic sampling seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    rows = read_jsonl(input_path)
    selected_native, diagnostics = choose_subset(rows, int(args.size), int(args.seed))
    compatible = [to_compatible_record(row, index) for index, row in enumerate(selected_native, start=1)]

    native_path = out_dir / "mentions_external_test_120_native.jsonl"
    compatible_path = out_dir / "mentions_external_test_120.jsonl"
    summary_path = out_dir / "external_test_120_summary.json"

    write_jsonl(native_path, selected_native)
    write_jsonl(compatible_path, compatible)
    summary = {
        "created_from": str(input_path),
        "outputs": {
            "native": str(native_path),
            "compatible": str(compatible_path),
            "summary": str(summary_path),
        },
        "sampling": diagnostics,
        "source_summary": summarize(rows),
        "native_subset_summary": summarize(selected_native),
        "compatible_subset_summary": summarize(compatible),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary["compatible_subset_summary"], ensure_ascii=False, indent=2))
    print(f"[done] {compatible_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
