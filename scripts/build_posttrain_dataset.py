#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from pathlib import Path


SEED = 20260620

NAMES = ["Mina", "Jonah", "Priya", "Luis", "Hana", "Omar", "Elena", "Noah", "Rafi", "Maya", "Sofia", "Nia", "Leo"]
CITIES = ["Boston", "Lima", "Seoul", "Oslo", "Nairobi", "Lisbon", "Toronto", "Kyoto", "Helsinki", "Valencia"]
OBJECTS = ["receipt", "notebook", "blue mug", "library card", "train ticket", "grocery bag", "garden glove", "bus pass", "tea cup", "label", "camera lens"]
PLACES = ["kitchen", "library", "museum", "market", "train station", "guest room", "courtyard", "bookshop", "clinic", "studio", "studio hallway"]
EVENTS = ["book club", "family lunch", "ticket pickup", "gallery walk", "market errand", "study session", "team standup", "maintenance check"]
MOODS = ["calm", "busy", "curious", "careful", "hopeful", "tired", "focused", "relieved"]
DOMAINS = ["home planning", "travel", "food", "arts", "customer support", "education", "community events", "personal organization", "retail", "reading", "weather planning"]
ADJECTIVES = ["quiet", "bright", "rainy", "early", "crowded", "warm", "formal", "casual", "shared", "local", "weekly", "small"]
DAYS = list(range(1, 29))
HOURS = list(range(8, 20))

FORMAT_CATALOG = [
    "json",
    "yaml",
    "one_word",
    "two_bullets",
    "csv",
    "label_value",
]

ARCHIVE_BIAS = [
    "science",
    "math",
    "engineering",
    "physics",
]


def with_request_id(idx: int, prompt: str) -> str:
    return f"Request {idx:05d}. {prompt}"


def row_json(*, idx: int, user: str, assistant: str, task: str | None = None, interrupt: bool = False) -> dict:
    payload: dict[str, object] = {
        "output": [
            {"role": "user", "content": with_request_id(idx, user)},
            {"role": "assistant", "content": assistant},
        ]
    }
    if task:
        payload["task"] = task
    if interrupt:
        payload["allow_user_interrupts"] = True
    return payload


def sample_from(pool: list[str], rng: random.Random) -> str:
    return pool[rng.randrange(len(pool))]


def text_format_rows(rng: random.Random, start: int, count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(start, start + count):
        name = sample_from(NAMES, rng)
        city = sample_from(CITIES, rng)
        day = sample_from(DAYS, rng)
        place = sample_from(PLACES, rng)
        obj = sample_from(OBJECTS, rng)
        ref = 1_000 + i
        fmt = sample_from(FORMAT_CATALOG, rng)
        if fmt == "json":
            user = f"Return JSON only with keys person and city. Note {ref}: {name} is heading to {city} on July {day} from {place}."
            assistant = json.dumps({"person": name, "city": city}, ensure_ascii=False)
            task = "format_json"
        elif fmt == "yaml":
            user = f"Return YAML only with keys task and place. Note {ref}: {name} is at {place} for a meeting."
            assistant = f"task: {name}\nplace: {place}"
            task = "format_yaml"
        elif fmt == "one_word":
            user = f"Answer with exactly one lowercase word for note {ref}: is the package count for {name} on {day} positive?"
            assistant = "yes"
            task = "format_one_word"
        elif fmt == "two_bullets":
            user = f"Give exactly two bullet points for preparing a {obj} for a {name} visit on {day}."
            assistant = f"- Prepare a clear list.\n- Bring the {obj}."
            task = "format_bullets"
        elif fmt == "csv":
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
            user = f"Return CSV with columns item,count for note {ref}: {a} notebooks and {b} receipts."
            assistant = f"item,count\nnotebook,{a}\nreceipt,{b}"
            task = "format_csv"
        else:
            val = sample_from(MOODS, rng)
            user = f"Reply as mood=value only. In note {ref}, the current mood is {val}."
            assistant = f"mood={val}"
            task = "format_label_value"
        rows.append(row_json(idx=i, user=user, assistant=assistant, task=task))
    return rows


def extraction_rows(rng: random.Random, start: int, count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(start, start + count):
        name = sample_from(NAMES, rng)
        city = sample_from(CITIES, rng)
        obj = sample_from(OBJECTS, rng)
        place = sample_from(PLACES, rng)
        ref = 2_000 + i
        user = f"Extract person, object, location from note {ref}: {name} left a {obj} at {place} while travelling to {city}."
        assistant = f"person: {name}\nobject: {obj}\nplace: {place}"
        rows.append(row_json(idx=i, user=user, assistant=assistant, task="extraction"))
    return rows


def control_and_memory_rows(rng: random.Random, start: int, count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(start, start + count):
        name = sample_from(NAMES, rng)
        city = sample_from(CITIES, rng)
        obj = sample_from(OBJECTS, rng)
        ref = 3_000 + i
        if i % 3 == 0:
            user = f"Repeat this exact code word only: code-{i:04d}. Do not explain."
            assistant = f"code-{i:04d}"
            task = "short_recall"
        elif i % 3 == 1:
            user = f"For note {ref}, write one lowercase word only and do not add anything else: is {obj} in {city}?"
            assistant = "yes"
            task = "format_one_word"
        else:
            user = f"Remember this sentence for later: {name} goes to {city} on July {sample_from(DAYS, rng)}."
            assistant = f"I will remember: {name} goes to {city}."
            task = "delayed_format_recall"
        rows.append(row_json(idx=i, user=user, assistant=assistant, task=task))
    return rows


def arithmetic_and_classification_rows(rng: random.Random, start: int, count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(start, start + count):
        name = sample_from(NAMES, rng)
        a = rng.randint(1, 8)
        b = rng.randint(0, 8)
        place = sample_from(PLACES, rng)
        hour = sample_from(HOURS, rng)
        ref = 4_000 + i
        if i % 2 == 0:
            user = f"Count math: In {name}'s plan on July {ref % 28 + 1}, add {a} notes and remove {b}. Give one integer."
            assistant = str(a + b)
            task = "arithmetic"
        else:
            mode = sample_from(["praise", "complaint", "question", "neutral"], rng)
            prompt = f"The package with {obj_phrase(a, b)} arrived at {place} by {hour}:00."
            user = f"Classify tone for note {ref}: {prompt}"
            assistant = mode
            task = "classification"
        rows.append(row_json(idx=i, user=user, assistant=assistant, task=task))
    return rows


def obj_phrase(a: int, b: int) -> str:
    return f"{a} + {b} items"


def interruption_rows(rng: random.Random, start: int, count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(start, start + count):
        name = sample_from(NAMES, rng)
        place = sample_from(PLACES, rng)
        ref = 5_000 + i
        rows.append(
            row_json(
                idx=i,
                user=f"Actually, stop and answer the new question for note {ref}: where is {name} going?",
                assistant=f"{name} is going to {place}.",
                task="interrupt",
                interrupt=True,
            )
        )
    return rows


def architecture_control_rows(rng: random.Random, start: int, count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(start, start + count):
        phrase = sample_from(ARCHIVE_BIAS, rng)
        if i % 2 == 0:
            user = f"Explain the concept in one sentence for note {10000 + i}: define {phrase} using a daily-life analogy."
            assistant = f"{phrase.title()} is best explained by breaking it into small repeated steps."
            task = "concept_lite"
        else:
            user = f"Describe one practical operational hint for {phrase} in a household context."
            assistant = f"Use one simple routine and avoid heavy tooling."
            task = "concept_lite"
        rows.append(row_json(idx=i, user=user, assistant=assistant, task=task))
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        key = hashlib.sha1(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def add_ids(rows: list[dict], start_idx: int = 1) -> list[dict]:
    out = []
    for idx, row in enumerate(rows, start=start_idx):
        if isinstance(row, dict):
            row.setdefault("id", f"posttrain_{idx:05d}")
            out.append(row)
    return out


def build_rows(count: int) -> list[dict]:
    count = max(1, int(count))
    rng = random.Random(SEED)
    rows: list[dict] = []
    total = count
    allocations = {
        "format": max(1, int(0.28 * total)),
        "extract": max(1, int(0.18 * total)),
        "memory": max(1, int(0.18 * total)),
        "reason": max(1, int(0.18 * total)),
        "interrupt": max(1, int(0.10 * total)),
        "concept": 0,
    }
    used = sum(allocations.values())
    allocations["concept"] = max(0, total - used)

    idx = 0
    rows.extend(text_format_rows(rng, idx, allocations["format"]))
    idx += allocations["format"]
    rows.extend(extraction_rows(rng, idx, allocations["extract"]))
    idx += allocations["extract"]
    rows.extend(control_and_memory_rows(rng, idx, allocations["memory"]))
    idx += allocations["memory"]
    rows.extend(arithmetic_and_classification_rows(rng, idx, allocations["reason"]))
    idx += allocations["reason"]
    rows.extend(interruption_rows(rng, idx, allocations["interrupt"]))
    idx += allocations["interrupt"]
    rows.extend(architecture_control_rows(rng, idx, allocations["concept"]))

    deduped = dedupe_rows(rows)
    # Keep ordering consistent for reproducibility after deduplication if collisions occur.
    return add_ids(deduped[:count], start_idx=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diverse post-train JSONL data.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("data/datasets/propagator_posttrain_generated.jsonl"))
    args = parser.parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(int(args.rows))
    with output.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
