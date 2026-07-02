#!/usr/bin/env python3
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_LINK = PROJECT_ROOT / "logs" / "train.latest.log"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "propagator-multimodal_1b"
METRICS_PATH = OUTPUT_ROOT / "train_metrics.jsonl"
PLOT_PATH = OUTPUT_ROOT / "train_loss.png"
LINE_RE = re.compile(
    r"\[Train\]\s+step=(?P<step>\d+)/(?P<total>\d+),\s+loss=(?P<loss>[-+0-9.eE]+),\s+"
    r"steps_per_sec=(?P<sps>[-+0-9.eE]+),\s+interval_steps_per_sec=(?P<isps>[-+0-9.eE]+)"
)


def read_existing() -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not METRICS_PATH.exists():
        return records
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                step = int(record.get("step", 0))
                if step > 0:
                    records[step] = record
            except Exception:
                continue
    return records


def parse_log() -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not LOG_LINK.exists():
        return records
    with LOG_LINK.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = LINE_RE.search(line)
            if not match:
                continue
            step = int(match.group("step"))
            records[step] = {
                "step": step,
                "total_steps": int(match.group("total")),
                "train_loss": float(match.group("loss")),
                "steps_per_sec": float(match.group("sps")),
                "interval_steps_per_sec": float(match.group("isps")),
                "time": time.time(),
                "source": str(LOG_LINK),
            }
    return records


def write_records(records: dict[int, dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = METRICS_PATH.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for step in sorted(records):
            f.write(json.dumps(records[step], ensure_ascii=False) + "\n")
    tmp_path.replace(METRICS_PATH)


def write_plot(records: dict[int, dict]) -> None:
    if not records:
        return
    steps = sorted(records)
    values = np.asarray([float(records[step]["train_loss"]) for step in steps], dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return
    fill = float(np.nanmean(values[finite]))
    values = np.where(finite, values, fill)
    with plt.rc_context(
        {
            "font.family": "monospace",
            "font.monospace": ["Roboto Mono", "DejaVu Sans Mono", "Liberation Mono", "monospace"],
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    ):
        plt.figure(figsize=(10, 4), facecolor="white")
        ax = plt.gca()
        ax.set_facecolor("white")
        plt.plot(steps, values, color="silver", linewidth=2, alpha=0.95, label="raw")
        if len(values) > 1:
            window = min(len(values), max(5, len(values) // 20))
            kernel = np.ones(window, dtype=np.float32) / float(window)
            smooth = np.convolve(values, kernel, mode="same")
            plt.plot(steps, smooth, color="black", linewidth=2, label=f"rolling mean ({window})")
        plt.title(f"Train weighted CE - Step {steps[-1]}")
        plt.xlabel("step")
        plt.ylabel("Train weighted CE")
        plt.grid(True, color="silver", alpha=0.35, linewidth=0.8)
        plt.legend(loc="best", frameon=False)
        plt.tight_layout()
        PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_PATH, dpi=120, facecolor="white")
        plt.close()


def sync_once() -> int:
    records = read_existing()
    records.update(parse_log())
    write_records(records)
    write_plot(records)
    return len(records)


def main() -> None:
    while True:
        sync_once()
        time.sleep(30)


if __name__ == "__main__":
    main()
