#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


NAMES = [
    "amber",
    "cobalt",
    "delta",
    "ember",
    "frost",
    "garnet",
    "harbor",
    "indigo",
    "jade",
    "kepler",
]

TOPICS = [
    ("matrix memory", "a fixed-size associative matrix that stores key-value traces"),
    ("delta update", "an error-correcting write that moves a stored value toward the new target"),
    ("streaming inference", "processing one chunk at a time while carrying recurrent memory forward"),
    ("turn taking", "predicting whether to keep listening, end the user turn, or start the model response"),
    ("audio tokens", "codec codebook ids carried beside text and protocol tokens"),
    ("RMSNorm", "normalizing by root mean square to stabilize hidden activations"),
    ("SwiGLU", "a gated feed-forward path that keeps useful channels and suppresses weak ones"),
    ("grouped memory reads", "splitting associative keys into groups so each group can retrieve a different value slice"),
    ("RoPE-style phase", "rotating memory keys by stream position so order is visible without a KV cache"),
    ("quantization", "storing trained weights in fewer bits for edge serving"),
]

EDGE_FACTS = [
    "Use a compact preset before quantization; quantization reduces weights but not every runtime buffer.",
    "The recurrent matrix is constant-size per layer, so serving memory does not grow linearly with prompt length.",
    "A 4-bit export should still reserve space for embeddings, recurrent state, temporary logits, and framework overhead.",
    "For a 2 GB target, prefer smaller hidden size, fewer layers, and grouped memory before relying on quantization alone.",
]

def row(user: str, assistant: str, *, interrupt: bool = False) -> dict:
    result = {
        "output": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
    if interrupt:
        result["allow_user_interrupts"] = True
    return result


def build_rows(count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(count):
        kind = i % 20
        name = NAMES[i % len(NAMES)]
        topic, definition = TOPICS[i % len(TOPICS)]
        other_topic, other_definition = TOPICS[(i * 7 + 3) % len(TOPICS)]

        if kind == 0:
            rows.append(row("What is your name?", "My name is Propagator."))
        elif kind == 1:
            rows.append(row("In one sentence, what is Propagator?", "Propagator is a streaming language and speech architecture that carries context in fixed-size recurrent matrix memory instead of a growing KV cache."))
        elif kind == 2:
            rows.append(row(f"Explain {topic} for this model.", f"In Propagator, {topic} means {definition}."))
        elif kind == 3:
            rows.append(row(f"Compare {topic} and {other_topic}.", f"{topic} is {definition}, while {other_topic} is {other_definition}."))
        elif kind == 4:
            rows.append(row(f"The code word is {name}. Repeat only the code word.", name))
        elif kind == 5:
            rows.append(row(f"Remember this setting: memory key group {i % 8}. What group did I give?", f"You gave memory key group {i % 8}."))
        elif kind == 6:
            rows.append(row("During user input, should the model answer immediately?", "No. During user input it should usually predict [LISTEN] and update memory until the turn boundary is clear."))
        elif kind == 7:
            rows.append(row("What should happen after [USER_END]?", "The model should transition into response mode, usually through [MODEL] and then an output modality token such as [TEXT_OUT] or [AUDIO_OUT]."))
        elif kind == 8:
            rows.append(row("Why keep a matrix memory instead of a KV cache?", "A matrix memory gives constant-size recurrent state per layer, which is useful for streaming and edge serving, though it compresses history lossily."))
        elif kind == 9:
            rows.append(row("Give a short checklist for edge serving.", "Choose the compact architecture, quantize weights, budget recurrent state, cap batch size, and test generation latency on the target device."))
        elif kind == 10:
            fact = EDGE_FACTS[i % len(EDGE_FACTS)]
            rows.append(row("Give one practical SL2610 deployment note.", fact))
        elif kind == 11:
            rows.append(row("How does the delta rule update memory?", "It reads the old value, computes the error against the target value, and writes a scaled outer-product correction into the matrix."))
        elif kind == 12:
            rows.append(row("Why add grouped associative reads?", "Grouped reads let separate key groups retrieve separate value slices, improving capacity without introducing token-length attention state."))
        elif kind == 13:
            rows.append(row("Why add a RoPE-style phase to recurrent memory keys?", "It gives read and write keys an order-dependent phase, so the fixed matrix can encode position signals without storing every token."))
        elif kind == 14:
            rows.append(row("What does SwiGLU improve?", "SwiGLU gives the feed-forward path a multiplicative gate, often improving useful channel selection for the same recurrent interface."))
        elif kind == 15:
            rows.append(row("What is the role of MoE here?", "MoE routes each step through a small set of feed-forward experts, increasing conditional capacity while keeping the recurrent memory design unchanged."))
        elif kind == 16:
            rows.append(row("What language should this post-training set emphasize?", "This post-training set is English-only and should concentrate on English instruction following, recall, and streaming control."))
        elif kind == 17:
            rows.append(row("Answer in exactly three words: fixed matrix memory", "fixed matrix memory"))
        elif kind == 18:
            rows.append(row("Summarize the README goal in one sentence.", "The goal is a streaming multimodal model that stores context in persistent matrix memory for constant-size inference state."))
        else:
            rows.append(row("Actually, stop and answer the new question: what are you?", "I am Propagator, a recurrent matrix-memory streaming model.", interrupt=True))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Propagator post-training JSONL data.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("data/propagator_posttrain_10k.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(max(1, int(args.rows)))
    with args.output.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
