#!/usr/bin/env python3
"""Plot Figure 4.2 training reward curves from JSON training logs.

Expected log format:
- List[dict] with keys: episode, ep_reward
- Or dict with key train_rewards: List[float]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


VECTOR_FORMATS = {"pdf", "svg", "eps"}
RASTER_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff", "webp"}


def parse_curve_spec(spec: str) -> Tuple[str, Path]:
    """Parse 'Label=path/to/log.json' into tuple."""
    if "=" not in spec:
        raise ValueError(f"Invalid --curve value: {spec}. Expected format: Label=path")
    label, path_str = spec.split("=", 1)
    label = label.strip()
    path = Path(path_str).expanduser().resolve()
    if not label:
        raise ValueError(f"Curve label cannot be empty: {spec}")
    if not path.exists():
        raise FileNotFoundError(f"Curve file not found: {path}")
    return label, path


def load_rewards(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        rewards = [float(item["ep_reward"]) for item in data if isinstance(item, dict) and "ep_reward" in item]
    elif isinstance(data, dict) and "train_rewards" in data:
        rewards = [float(x) for x in data["train_rewards"]]
    else:
        raise ValueError(f"Unsupported reward log format: {path}")

    if not rewards:
        raise ValueError(f"No rewards found in: {path}")
    return np.asarray(rewards, dtype=np.float64)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DRL training reward curves (Figure 4.2)")
    parser.add_argument(
        "--curve",
        action="append",
        required=True,
        help="Curve spec: Label=/absolute/or/relative/path/to/log.json (repeatable)",
    )
    parser.add_argument("--window", type=int, default=5, help="Moving-average window (episodes)")
    parser.add_argument(
        "--out",
        type=str,
        default="results/figures/figure_4_2_training_rewards.pdf",
        help="Output figure path (recommended vector formats: .pdf/.svg/.eps)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=sorted(VECTOR_FORMATS | RASTER_FORMATS),
        help="Force output format. If omitted, infer from --out extension.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs (e.g., png)")
    parser.add_argument("--title", type=str, default="Figure 4.2 Training Reward Curves of Three DRL Algorithms")
    parser.add_argument("--xlabel", type=str, default="Training Episode")
    parser.add_argument("--ylabel", type=str, default="Cumulative Reward (Moving Average)")
    args = parser.parse_args()

    curves: List[Tuple[str, np.ndarray]] = []
    max_len = 0
    for spec in args.curve:
        label, path = parse_curve_spec(spec)
        rewards = load_rewards(path)
        curves.append((label, rewards))
        max_len = max(max_len, len(rewards))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    for label, rewards in curves:
        smooth = moving_average(rewards, args.window)
        x = np.arange(1, len(smooth) + 1)
        ax.plot(x, smooth, linewidth=2, label=label)

    ax.set_title(args.title, fontsize=14)
    ax.set_xlabel(args.xlabel, fontsize=12)
    ax.set_ylabel(args.ylabel, fontsize=12)
    ax.set_xlim(1, max_len)
    ax.legend(frameon=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use embeddable TrueType fonts in vector outputs for better thesis compatibility.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    output_format = args.format or out_path.suffix.lower().lstrip(".")
    if not output_format:
        output_format = "pdf"

    fig.tight_layout()
    if output_format in VECTOR_FORMATS:
        fig.savefig(out_path, format=output_format, bbox_inches="tight")
    else:
        fig.savefig(out_path, format=output_format, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
