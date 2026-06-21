#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[a-z0-9']+")
FORBIDDEN = ["SL2610", "224x224", "camera frame"]
TOPIC_KEYWORDS = {
    "architecture": {"propagator", "matrix", "memory", "quantization", "streaming", "codec", "token", "moe", "rope", "rmsnorm", "swiglu"},
    "stem": {"sensor", "robot", "motor", "pressure", "gauge", "temperature", "conveyor", "valve", "pump", "science", "scientific"},
    "home": {"kitchen", "guest", "room", "picnic", "laundry", "sofa", "table", "umbrella", "garden", "meal"},
    "travel": {"city", "trip", "ticket", "station", "visit", "travel", "luggage", "hotel"},
    "arts_culture": {"museum", "book", "library", "painting", "music", "novel", "tour"},
    "service_retail": {"delivery", "receipt", "store", "shelf", "customer", "package", "market"},
    "planning_admin": {"meeting", "calendar", "schedule", "event", "checklist", "plan", "reminder"},
}


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def shingles(tok: list[str], n: int = 5) -> set[tuple[str, ...]]:
    if len(tok) < n:
        return {tuple(tok)} if tok else set()
    return {tuple(tok[i : i + n]) for i in range(len(tok) - n + 1)}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json_like(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    data = json.loads(content)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def row_text(row: dict[str, Any]) -> tuple[str, str]:
    if isinstance(row.get("output"), list):
        user_parts = []
        assistant_parts = []
        for msg in row["output"]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "")
            content = str(msg.get("content") or msg.get("value") or "")
            if role == "user":
                user_parts.append(content)
            elif role in {"assistant", "model", "gpt"}:
                assistant_parts.append(content)
        return "\n".join(user_parts), "\n".join(assistant_parts)
    user = str(row.get("question") or row.get("prompt") or row.get("image_text") or "")
    assistant = str(row.get("answer") or row.get("response") or row.get("label") or "")
    return user, assistant


def topic_counts(texts: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        tok = set(tokens(text))
        matched = False
        for topic, words in TOPIC_KEYWORDS.items():
            if tok & words:
                counts[topic] += 1
                matched = True
        if not matched:
            counts["general_other"] += 1
    return counts


def analyze_file(path: Path, max_pairs: int) -> dict[str, Any]:
    rows = read_json_like(path)
    user_texts = [row_text(row)[0] for row in rows]
    # ignore empty user prompts to avoid false positives
    user_texts = [text for text in user_texts if text]
    assistant_texts = [row_text(row)[1] for row in rows]
    normalized = [normalize(text) for text in user_texts]
    counts = Counter(normalized)
    exact_duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
    tok_lists = [tokens(text) for text in user_texts]
    total_tokens = sum(len(toks) for toks in tok_lists)
    unique_tokens = len({tok for toks in tok_lists for tok in toks})
    lexical_diversity = unique_tokens / max(1, total_tokens)

    shingle_sets = [shingles(toks) for toks in tok_lists]
    near_pairs = 0
    comparisons = 0
    examples = []
    limit = min(len(shingle_sets), max_pairs)
    for i in range(limit):
        for j in range(i + 1, limit):
            comparisons += 1
            score = jaccard(shingle_sets[i], shingle_sets[j])
            if score >= 0.82 and normalized[i] != normalized[j]:
                near_pairs += 1
                if len(examples) < 5:
                    examples.append({"i": i, "j": j, "score": round(score, 3), "a": user_texts[i], "b": user_texts[j]})

    forbidden_hits = {
        phrase: sum(1 for text in user_texts + assistant_texts if phrase.lower() in text.lower())
        for phrase in FORBIDDEN
    }
    topics = topic_counts(user_texts)
    task_counts = Counter(str(row.get("task") or "<none>") for row in rows)
    top_prompts = [
        {"count": count, "text": text}
        for text, count in Counter(user_texts).most_common(5)
        if count > 1
    ]
    return {
        "path": str(path),
        "rows": len(rows),
        "valid_user_rows": len(user_texts),
        "unique_user_prompts": len(counts),
        "exact_duplicate_user_rows": exact_duplicate_rows,
        "lexical_diversity": round(lexical_diversity, 4),
        "near_duplicate_pairs_checked": comparisons,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_examples": examples,
        "forbidden_hits": forbidden_hits,
        "topic_counts": dict(topics),
        "task_counts_top": dict(task_counts.most_common(12)),
        "top_repeated_prompts": top_prompts,
    }


def cross_file_coverage(paths: list[Path], max_pairs: int, threshold: float = 0.82) -> dict[str, Any]:
    sources: list[tuple[str, list[tuple[int, str, str]]]] = []
    for path in paths:
        rows = read_json_like(path)
        user_texts = []
        for idx, row in enumerate(rows):
            user_text, _ = row_text(row)
            if not user_text:
                continue
            norm = normalize(user_text)
            if not norm:
                continue
            user_texts.append((idx, norm, row.get("task") or "<none>"))
        if user_texts:
            sources.append((str(path), user_texts))

    if len(sources) < 2:
        return {
            "evaluated_file_pairs": 0,
            "evaluated_row_pairs": 0,
            "near_duplicate_pairs": 0,
            "examples": [],
        }

    cap = max(1, max_pairs)
    total_row_pairs = 0
    near_pairs = 0
    examples: list[dict[str, Any]] = []

    for i in range(len(sources)):
        left_path, left_rows = sources[i]
        left_rows = left_rows[:cap]
        left_shingles = [(idx, text, task, shingles(normalize(text).split(), 5)) for idx, text, task in left_rows]
        for j in range(i + 1, len(sources)):
            right_path, right_rows = sources[j]
            right_rows = right_rows[:cap]
            right_shingles = [(idx, text, task, shingles(normalize(text).split(), 5)) for idx, text, task in right_rows]
            for li, ltext, ltask, lset in left_shingles:
                for ri, rtext, rtask, rset in right_shingles:
                    total_row_pairs += 1
                    score = jaccard(lset, rset)
                    if score >= threshold and ltext != rtext:
                        near_pairs += 1
                        if len(examples) < 10:
                            examples.append(
                                {
                                    "left": left_path,
                                    "right": right_path,
                                    "left_index": li,
                                    "right_index": ri,
                                    "left_task": ltask,
                                    "right_task": rtask,
                                    "score": round(score, 3),
                                    "left_prompt": ltext,
                                    "right_prompt": rtext,
                                }
                            )

    return {
        "evaluated_file_pairs": len(sources) * (len(sources) - 1) // 2,
        "evaluated_row_pairs": total_row_pairs,
        "near_duplicate_pairs": near_pairs,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local JSONL prompt diversity and forbidden prompt artifacts.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    paths = args.paths or [
        REPO_ROOT / "data" / "propagator_instruction_balanced_seed.jsonl",
        REPO_ROOT / "data" / "propagator_image_recognition_seed.jsonl",
        REPO_ROOT / "data" / "propagator_posttrain_10k.jsonl",
        REPO_ROOT / "data" / "propagator_identity.jsonl",
    ]
    failed = False
    resolved = [path if path.is_absolute() else REPO_ROOT / path for path in paths]
    for path in resolved:
        report = analyze_file(path, args.max_pairs)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if any(report["forbidden_hits"].values()):
            failed = True
        if report["rows"] and report["exact_duplicate_user_rows"] / report["rows"] > 0.20:
            failed = True
        arch_share = report["topic_counts"].get("architecture", 0) / max(1, report["rows"])
        stem_share = report["topic_counts"].get("stem", 0) / max(1, report["rows"])
        if arch_share > 0.25 or stem_share > 0.25:
            failed = True
    cross = cross_file_coverage(resolved, max_pairs=args.max_pairs)
    print("[cross-file-summary]", json.dumps(cross, ensure_ascii=False, indent=2))
    if cross["near_duplicate_pairs"] > 0:
        failed = True
    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
