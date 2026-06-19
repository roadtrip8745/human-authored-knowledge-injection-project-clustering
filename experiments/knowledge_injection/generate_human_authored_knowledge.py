# -*- coding: utf-8 -*-
"""
Generate human-authored knowledge files for the public reproducibility package.

The output is stored as Markdown because Markdown is easy to read, edit, diff,
and review. The method itself is not tied to Markdown; any human-editable
structured note format could carry the same canonical names, aliases, and
cannot-merge hints.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MENTIONS = r"data\synthetic\extraction_run_20260618_085129\mentions_extracted.jsonl"
DEFAULT_OUT_DIR = r"data\synthetic\human_authored_knowledge"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no} is not valid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def pick_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def unique(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if limit is not None and len(result) >= limit:
            break
    return result


def group_mentions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        project_id = pick_text(row, "gold_project_id", "gold_cluster", "project_id", "cluster_id")
        if project_id:
            grouped[project_id].append(row)
    return dict(sorted(grouped.items()))


def canonical_for(rows: list[dict[str, Any]], fallback: str) -> str:
    candidates = [
        pick_text(row, "canonical_project", "gold_project", "project_name", "canonical")
        for row in rows
    ]
    counts = Counter(value for value in candidates if value)
    if counts:
        return counts.most_common(1)[0][0]
    return fallback


def aliases_for(rows: list[dict[str, Any]], canonical: str, limit: int = 8) -> list[str]:
    candidates: list[str] = [canonical]
    for row in rows:
        mention = pick_text(row, "mention_text", "mention", "text", "project_mention")
        extracted = pick_text(row, "extracted_project", "extracted_text")
        candidates.extend([mention, extracted])
        for key in ("aliases_used", "expected_mentions", "aliases"):
            value = row.get(key)
            if isinstance(value, list):
                candidates.extend(str(item) for item in value if item)
    return unique(candidates, limit=limit)


def write_knowledge(path: Path, title: str, projects: dict[str, list[dict[str, Any]]], *, coverage: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(projects.items())[: max(1, round(len(projects) * coverage))]
    lines = [
        f"# {title}",
        "",
        "> Synthetic human-authored knowledge for reproducibility.",
        "> The file is Markdown only for readability and editability.",
        "> It contains no real company, person, or project information.",
        "",
    ]
    for project_id, rows in selected:
        canonical = canonical_for(rows, project_id)
        aliases = aliases_for(rows, canonical)
        lines.extend(
            [
                f"## {canonical}",
                "",
                f"- project_id_hint: {project_id}",
                "- confidence: synthetic",
                "- aliases:",
            ]
        )
        for alias in aliases:
            lines.append(f"  - {alias}")
        lines.extend(
            [
                "- cannot_merge_with: []",
                "- notes:",
                "  - Maintained as editable project knowledge for clustering experiments.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, projects: dict[str, list[dict[str, Any]]]) -> None:
    payload = {
        "source": DEFAULT_MENTIONS,
        "project_count": len(projects),
        "outputs": {
            "full": "data/synthetic/human_authored_knowledge/project_knowledge_full.md",
            "partial": "data/synthetic/human_authored_knowledge/project_knowledge_partial.md",
            "noisy": "data/synthetic/human_authored_knowledge/project_knowledge_noisy.md",
        },
        "note": "This public generator creates readable Markdown notes from synthetic mention records.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic human-authored knowledge files.")
    parser.add_argument("--mentions", default=DEFAULT_MENTIONS, help="Path to mentions_extracted.jsonl.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for knowledge files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(Path(args.mentions))
    projects = group_mentions(rows)
    if not projects:
        raise ValueError("No project groups were found in the mention records.")

    out_dir = Path(args.out_dir)
    write_knowledge(out_dir / "project_knowledge_full.md", "Full Human-Authored Knowledge", projects, coverage=1.0)
    write_knowledge(out_dir / "project_knowledge_partial.md", "Partial Human-Authored Knowledge", projects, coverage=0.67)
    write_knowledge(out_dir / "project_knowledge_noisy.md", "Noisy Human-Authored Knowledge", projects, coverage=0.67)
    write_summary(out_dir / "knowledge_generation_summary.json", projects)
    print(f"[done] wrote human-authored knowledge files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
