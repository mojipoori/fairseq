#!/usr/bin/env python3

"""Extract Fairseq epoch metrics across multiple seeds and plot per seed."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# PATHS
# ============================================================

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "checkpoints_of_seeds"
OUT_DIR = ROOT / "plots"


# ============================================================
# SEEDS
# ============================================================

SEEDS = [
    "seed1993",
    "seed1402",
    "seed2025",
]


# ============================================================
# RUNS
# ============================================================

RUNS = [
    {
        "label": "Emoji (α=1, Proposed)",
        "color": "#1F77B4",
        "folder": "iwslt14_emoji",
    },
    {
        "label": "AdaBelief",
        "color": "#D62728",
        "folder": "iwslt14_adabelief",
    },
    {
        "label": "AdamW",
        "color": "#9467BD",
        "folder": "iwslt14_adamw",
    },
    {
        "label": "Lion",
        "color": "#E377C2",
        "folder": "iwslt14_lion",
    },
    {
        "label": "Emoji (α=0, Without Centering)",
        "color": "#FF7F0E",
        "folder": "iwslt14_emojiabl",
    },
]


# ============================================================
# PLOT CONFIGURATION
# ============================================================

PLOTS = [
    {
        "key": "train_loss",
        "ylabel": "Train Loss",
        "title": "Train Loss vs Epoch",
        "filename": "train_loss_vs_epoch",
    },
    {
        "key": "valid_loss",
        "ylabel": "Validation Loss",
        "title": "Validation Loss vs Epoch",
        "filename": "valid_loss_vs_epoch",
    },
    {
        "key": "valid_bleu",
        "ylabel": "Validation BLEU",
        "title": "Validation BLEU vs Epoch",
        "filename": "valid_bleu_vs_epoch",
    },
    {
        "key": "valid_ppl",
        "ylabel": "Validation PPL",
        "title": "Validation PPL vs Epoch",
        "filename": "valid_ppl_vs_epoch",
    },
]


# ============================================================
# FIGURE / SAVE SETTINGS
# ============================================================

FIGSIZE = (8.5, 5.5)

SAVE_FORMAT = "svg"
SAVE_DPI = 300
PAD_INCHES = 0.1

SAVE_SVG = True
SAVE_PNG = True

SHOW_PREVIEW = True
WAIT_FOR_PREVIEW_CLOSE = True

# Margins / spacing (same as GPT per-seed plotting script).
LEFT = 0.695
BOTTOM = 0.131
RIGHT = 0.996
TOP = 0.651
WSPACE = 0.2
HSPACE = 0.2


# ============================================================
# PARSE LOG
# ============================================================

def parse_log(path: Path) -> dict[str, dict[int, float]]:
    """Parse Fairseq JSON log lines into per-epoch metric series."""

    series: dict[str, dict[int, float]] = {
        "train_loss": {},
        "valid_loss": {},
        "valid_bleu": {},
        "valid_ppl": {},
    }

    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line.startswith("{"):
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = row.get("epoch")

            if epoch is None:
                continue

            epoch = int(epoch)

            for key in series:
                if key in row:
                    series[key][epoch] = float(row[key])

    return series


# ============================================================
# COLLECT RUN DATA
# ============================================================

def collect_run_data(
    run: dict,
) -> dict[str, dict[str, dict[int, float]]]:
    """
    Collect metric data for all seeds.

    Returns:
        metric -> seed -> epoch -> value
    """

    data: dict[str, dict[str, dict[int, float]]] = {
        "train_loss": {},
        "valid_loss": {},
        "valid_bleu": {},
        "valid_ppl": {},
    }

    for seed in SEEDS:

        log_path = (
            ROOT
            / seed
            / run["folder"]
            / "train.log"
        )

        if not log_path.exists():
            print(
                f"Warning: missing log for "
                f"{run['label']} ({seed}): "
                f"{log_path}"
            )
            continue

        parsed = parse_log(log_path)

        for metric in data:
            if parsed.get(metric):
                data[metric][seed] = parsed[metric]

    return data


# ============================================================
# SAVE HELPERS
# ============================================================

def _filename_base(plot_cfg: dict, seed: str) -> str:
    """Build the timestamped base filename for one seed plot."""
    stem = plot_cfg["filename"]
    if stem.endswith(".svg"):
        stem = stem[:-4]
    return f"{stem}_seed_{seed}_{RUN_TIMESTAMP}"


def save_figure_on_close(fig, plot_cfg: dict, seed: str):
    """Save timestamped SVG/PNG when the preview window is closed."""

    saved = {"done": False}
    filename_base = _filename_base(plot_cfg, seed)

    def save_on_close(event):
        if saved["done"]:
            return
        saved["done"] = True

        if SAVE_SVG:
            svg_path = OUT_DIR / f"{filename_base}.svg"
            fig.savefig(
                svg_path,
                format=SAVE_FORMAT,
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=PAD_INCHES,
            )
            print(f"Saved on close: {svg_path}")

        if SAVE_PNG:
            png_path = OUT_DIR / f"{filename_base}.png"
            fig.savefig(
                png_path,
                format="png",
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=PAD_INCHES,
            )
            print(f"Saved on close: {png_path}")

    fig.canvas.mpl_connect("close_event", save_on_close)


def save_figure_now(fig, plot_cfg: dict, seed: str):
    """Save the figure directly when previews are disabled."""

    filename_base = _filename_base(plot_cfg, seed)

    if SAVE_SVG:
        svg_path = OUT_DIR / f"{filename_base}.svg"
        fig.savefig(
            svg_path,
            format=SAVE_FORMAT,
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=PAD_INCHES,
        )
        print(f"Saved: {svg_path}")

    if SAVE_PNG:
        png_path = OUT_DIR / f"{filename_base}.png"
        fig.savefig(
            png_path,
            format="png",
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=PAD_INCHES,
        )
        print(f"Saved: {png_path}")


# ============================================================
# PLOT (ONE METRIC FOR ONE SEED)
# ============================================================

def plot_metric_for_seed(
    runs_data: list[tuple[dict, dict]],
    plot_cfg: dict,
    seed: str,
) -> bool:
    """Plot a single metric for a single seed (no averaging across seeds)."""

    fig, ax = plt.subplots(figsize=FIGSIZE)

    fig.subplots_adjust(
        left=LEFT,
        bottom=BOTTOM,
        right=RIGHT,
        top=TOP,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    plotted = False

    zorders = {
        "Emoji (α=1, Proposed)": 9,
        "Emoji (α=0, Without Centering)": 8,
        "AdamW": 7,
        "Emoji (ε=1e-16)": 6,
        "AdaBelief (ε=1e-16)": 5,
        "Emoji (ε=1e-12)": 4,
        "AdaBelief (ε=1e-12)": 3,
        "AdaBelief": 2,
        "Lion": 1,
    }

    for run, data in runs_data:

        seed_series = data.get(plot_cfg["key"], {})
        epoch_values = seed_series.get(seed, {})

        if not epoch_values:
            continue

        ordered = sorted(
            (int(e), float(v))
            for e, v in epoch_values.items()
            if v is not None and np.isfinite(v)
        )

        if not ordered:
            continue

        epochs = np.asarray([e for e, _ in ordered], dtype=int)
        values = np.asarray([v for _, v in ordered], dtype=float)

        ax.plot(
            epochs,
            values,
            color=run["color"],
            label=run["label"],
            linewidth=2.0,
            alpha=0.8,
            marker="o",
            markersize=3.5,
            markevery=max(1, len(epochs) // 20),
            zorder=zorders.get(run["label"], 1),
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Epoch")
    ax.set_ylabel(plot_cfg["ylabel"])
    ax.set_title(f"{plot_cfg['title']} (Seed {seed})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)

    if SHOW_PREVIEW:
        save_figure_on_close(fig, plot_cfg, seed)
        plt.show(block=WAIT_FOR_PREVIEW_CLOSE)
    else:
        save_figure_now(fig, plot_cfg, seed)

    plt.close(fig)

    return True


# ============================================================
# SUMMARY
# ============================================================

def print_summary(run: dict, data: dict) -> None:
    """Print number of seeds and epochs available."""

    print(f"{run['label']}:")

    for metric in (
        "train_loss",
        "valid_loss",
        "valid_bleu",
        "valid_ppl",
    ):
        seed_data = data.get(metric, {})
        print(f"  {metric}: {len(seed_data)} seeds")

        for seed, epochs in seed_data.items():
            print(f"    {seed}: {len(epochs)} epochs")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs_data = []

    for run in RUNS:
        data = collect_run_data(run)
        print_summary(run, data)
        runs_data.append((run, data))

    print()

    if not runs_data:
        raise SystemExit(
            "No train.log files found; nothing to plot."
        )

    for seed in SEEDS:
        print(f"\n--- {seed} ---")

        for plot_cfg in PLOTS:
            ok = plot_metric_for_seed(runs_data, plot_cfg, seed)

            if not ok:
                print(
                    f"Skipped {plot_cfg['title']} "
                    f"(seed {seed}): metric missing or no data"
                )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()