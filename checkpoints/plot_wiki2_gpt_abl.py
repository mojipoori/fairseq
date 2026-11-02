#!/usr/bin/env python3
"""
Plot Fairseq GPT-style Transformer metrics on WikiText-2.

The script:
1. Reads train.log files for three optimizers and three seeds.
2. Extracts epoch-wise training/validation loss and perplexity.
3. Plots mean +/- one sample standard deviation across seeds.
4. Saves SVG and PNG figures when each plot window is closed.
5. Exports epoch-wise metrics and final-10-epoch summaries to CSV.
6. Exports a compact table-style CSV for the final ten epochs.

This script does not start training or modify checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# ============================================================
# PATHS
# ============================================================

# Expected script location:
# <project_root>/fairseq/plot_wikitext2_gpt_metrics.py
REPO = Path(__file__).resolve().parents[1]

CHECKPOINT_ROOT = REPO / "checkpoints"
OUT_DIR = CHECKPOINT_ROOT / "wikitext2_gpt_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEEDS = [1993, 1402, 2025]

MODEL_ARCH = "transformer_lm_gpt"

LAYERS = 4
EMBED_DIM = 128
HEADS = 4
FFN_DIM = 512

DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1

WARMUP_UPDATES = 1000
SCHEDULER = "inverse_sqrt"

TOKENS_PER_SAMPLE = 128
MAX_TOKENS = 1024
UPDATE_FREQ = 4

LAST_N_EPOCHS = 10


OPTIMIZERS = [
    {
        "name": "emojiabl",
        "label": "Emoji (α=0, Without Centering)",
        "color": "#FF7F0E",
        "lr": "5e-4",
        "weight_decay": "0.01",
        "alphaa": 0,
    },
    {
        "name": "emoji",
        "label": "Emoji (α=1, Proposed)",
        "color": "#1F77B4",
        "lr": "5e-4",
        "weight_decay": "0.01",
    },
    {
        "name": "adabelief",
        "label": "AdaBelief",
        "color": "#D62728",
        "lr": "5e-4",
        "weight_decay": "0.01",
    },
]


# ============================================================
# PLOT CONFIGURATION
# ============================================================

METRICS = [
    {
        "key": "train_loss",
        "ylabel": "Training Loss",
        "title": "Training Loss vs. Epoch",
        "filename": "train_loss_vs_epoch",
    },
    {
        "key": "valid_loss",
        "ylabel": "Validation Loss",
        "title": "Validation Loss vs. Epoch",
        "filename": "valid_loss_vs_epoch",
    },
    {
        "key": "train_ppl",
        "ylabel": "Training Perplexity",
        "title": "Training Perplexity vs. Epoch",
        "filename": "train_ppl_vs_epoch",
    },
    {
        "key": "valid_ppl",
        "ylabel": "Validation Perplexity",
        "title": "Validation Perplexity vs. Epoch",
        "filename": "valid_ppl_vs_epoch",
    },
]

FIGSIZE = (8.5, 5.5)

# Save settings transferred from the LSTM plotting script.
SAVE_FORMAT = "svg"
SAVE_DPI = 300
PAD_INCHES = 0.1

# Preserve both output formats from the original GPT plotting script.
SAVE_SVG = True
SAVE_PNG = True

SHOW_PREVIEW = True
WAIT_FOR_PREVIEW_CLOSE = True

# Figure spacing transferred from the LSTM plotting script.
LEFT = 0.695
BOTTOM = 0.131
RIGHT = 0.996
TOP = 0.651
WSPACE = 0.2
HSPACE = 0.2


# ============================================================
# BUILD THE EXACT RUN DIRECTORY
# ============================================================

def build_config_name(optimizer: dict) -> str:
    """Reconstruct the configuration name used by the GPT run script."""

    config_name = (
        f"Arch{MODEL_ARCH}"
        f"_L{LAYERS}"
        f"_E{EMBED_DIM}"
        f"_H{HEADS}"
        f"_F{FFN_DIM}"
        f"_D{DROPOUT}"
        f"_AD{ATTENTION_DROPOUT}"
        f"_WD{optimizer['weight_decay']}"
        f"_WU{WARMUP_UPDATES}"
        f"_LR{optimizer['lr']}"
        f"_Sched{SCHEDULER}"
        f"_TPS{TOKENS_PER_SAMPLE}"
        f"_MT{MAX_TOKENS}"
        f"_UF{UPDATE_FREQ}"
    )

    if optimizer["name"] == "emojiabl":
        config_name += f"_Alpha{optimizer['alphaa']}"

    return config_name


def build_log_path(optimizer: dict, seed: int) -> Path:
    """Return the train.log path for one optimizer and seed."""

    config_name = build_config_name(optimizer)

    run_dir = (
        CHECKPOINT_ROOT
        / f"wikitext-2_{MODEL_ARCH}_{optimizer['name']}"
        / config_name
        / f"seed_{seed}"
    )

    return run_dir / "train.log"


# ============================================================
# LOG PARSING HELPERS
# ============================================================

def to_float(value):
    """Convert a scalar log value to float; return None if unavailable."""

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def get_epoch(row: dict):
    """Extract the epoch number from a Fairseq JSON record."""

    value = row.get("epoch")

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_record_tag(row: dict) -> str:
    """Extract a tag if the log format provides one."""

    for key in ("tag", "prefix", "subset"):
        value = row.get(key)

        if isinstance(value, str):
            return value.lower()

    return ""


def first_available(row: dict, keys: tuple[str, ...]):
    """Return the first usable scalar from a list of possible field names."""

    for key in keys:
        value = to_float(row.get(key))

        if value is not None:
            return value

    return None


def parse_log(path: Path) -> dict[str, dict[int, float]]:
    """
    Parse a Fairseq JSON log.

    Each metric maps epoch -> value. The latest usable epoch-level value
    is retained. Explicit train_inner records are ignored.
    """

    series = {
        "train_loss": {},
        "train_ppl": {},
        "valid_loss": {},
        "valid_ppl": {},
    }

    with path.open("r", encoding="utf-8", errors="ignore") as log_file:
        for line in log_file:
            line = line.strip()

            if not line.startswith("{"):
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(row, dict):
                continue

            epoch = get_epoch(row)

            if epoch is None:
                continue

            tag = get_record_tag(row)

            # Ignore explicitly identified mid-epoch inner records.
            if "train_inner" in tag:
                continue

            # Training metrics.
            train_loss = first_available(row, ("train_loss",))
            train_ppl = first_available(
                row,
                ("train_ppl", "train_perplexity"),
            )

            if tag == "train":
                if train_loss is None:
                    train_loss = first_available(row, ("loss",))

                if train_ppl is None:
                    train_ppl = first_available(
                        row,
                        ("ppl", "perplexity"),
                    )

            if train_loss is not None:
                series["train_loss"][epoch] = train_loss

            if train_ppl is not None:
                series["train_ppl"][epoch] = train_ppl

            # Validation metrics.
            valid_loss = first_available(
                row,
                ("valid_loss", "validation_loss"),
            )
            valid_ppl = first_available(
                row,
                ("valid_ppl", "valid_perplexity", "validation_ppl"),
            )

            if "valid" in tag:
                if valid_loss is None:
                    valid_loss = first_available(row, ("loss",))

                if valid_ppl is None:
                    valid_ppl = first_available(
                        row,
                        ("ppl", "perplexity"),
                    )

            if valid_loss is not None:
                series["valid_loss"][epoch] = valid_loss

            if valid_ppl is not None:
                series["valid_ppl"][epoch] = valid_ppl

    return series


# ============================================================
# COLLECT ALL RUNS
# ============================================================

def collect_all_runs():
    """
    Return:
    data[optimizer_name][seed][metric][epoch] = value
    """

    data = {}
    missing_logs = []
    metric_counts = defaultdict(int)

    for optimizer in OPTIMIZERS:
        optimizer_name = optimizer["name"]
        data[optimizer_name] = {}

        for seed in SEEDS:
            log_path = build_log_path(optimizer, seed)
            print(f"Checking: {log_path}")

            if not log_path.is_file():
                print(
                    f"WARNING: log not found for "
                    f"{optimizer['label']}, seed {seed}"
                )
                missing_logs.append(log_path)
                continue

            parsed = parse_log(log_path)
            data[optimizer_name][seed] = parsed

            for metric_name, epoch_values in parsed.items():
                metric_counts[(optimizer_name, metric_name)] += len(
                    epoch_values
                )

            print(
                f"Loaded {optimizer['label']} / seed {seed}: "
                f"{sum(len(values) for values in parsed.values())} "
                f"metric values"
            )

    print("\nMetric extraction summary:")

    for optimizer in OPTIMIZERS:
        for metric in METRICS:
            count = metric_counts[(optimizer["name"], metric["key"])]
            print(
                f"  {optimizer['label']:<32} "
                f"{metric['key']:<12} {count} epoch values"
            )

    if missing_logs:
        print("\nMissing log files:")
        for path in missing_logs:
            print(f"  {path}")

    return data


# ============================================================
# STATISTICS ACROSS SEEDS
# ============================================================

def epoch_statistics(
    data: dict,
    optimizer_name: str,
    metric_name: str,
):
    """Calculate epoch-wise mean and sample standard deviation across seeds."""

    all_epochs = sorted({
        epoch
        for seed_data in data.get(optimizer_name, {}).values()
        for epoch in seed_data.get(metric_name, {})
    })

    result = []

    for epoch in all_epochs:
        values = []

        for seed_data in data.get(optimizer_name, {}).values():
            value = seed_data.get(metric_name, {}).get(epoch)

            if value is not None and math.isfinite(value):
                values.append(float(value))

        if not values:
            continue

        result.append({
            "epoch": epoch,
            "n_seeds": len(values),
            "mean": float(np.mean(values)),
            "std": (
                float(np.std(values, ddof=1))
                if len(values) >= 2
                else float("nan")
            ),
        })

    return result


# ============================================================
# SAVE FIGURES WHEN THE PLOT WINDOW IS CLOSED
# ============================================================

def save_figure_on_close(fig, metric: dict):
    """Save SVG and PNG files with the run timestamp when the window closes."""

    saved = {"done": False}

    def save_on_close(event):
        if saved["done"]:
            return

        saved["done"] = True

        if SAVE_SVG:
            svg_path = OUT_DIR / f"{metric['filename']}_{RUN_TIMESTAMP}.svg"
            fig.savefig(
                svg_path,
                format=SAVE_FORMAT,
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=PAD_INCHES,
            )
            print(f"Saved on close: {svg_path}")

        if SAVE_PNG:
            png_path = OUT_DIR / f"{metric['filename']}_{RUN_TIMESTAMP}.png"
            fig.savefig(
                png_path,
                format="png",
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=PAD_INCHES,
            )
            print(f"Saved on close: {png_path}")

    fig.canvas.mpl_connect("close_event", save_on_close)


# ============================================================
# PLOT METRICS
# ============================================================

def plot_metric(data: dict, metric: dict):
    metric_name = metric["key"]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Use the exact margins and spacing from the LSTM plotting script.
    fig.subplots_adjust(
        left=LEFT,
        bottom=BOTTOM,
        right=RIGHT,
        top=TOP,
        wspace=WSPACE,
        hspace=HSPACE,
    )

    plotted_anything = False

    for optimizer in OPTIMIZERS:
        optimizer_name = optimizer["name"]

        stats = epoch_statistics(data, optimizer_name, metric_name)

        if not stats:
            print(
                f"WARNING: no {metric_name} values for "
                f"{optimizer['label']}"
            )
            continue

        epochs = np.asarray(
            [item["epoch"] for item in stats],
            dtype=int,
        )
        means = np.asarray(
            [item["mean"] for item in stats],
            dtype=float,
        )
        stds = np.asarray(
            [item["std"] for item in stats],
            dtype=float,
        )

        ax.plot(
            epochs,
            means,
            marker="o",
            markersize=2.5,
            linewidth=1.8,
            color=optimizer["color"],
            label=optimizer["label"],
        )

        valid_std = np.isfinite(stds)

        if valid_std.any():
            ax.fill_between(
                epochs[valid_std],
                means[valid_std] - stds[valid_std],
                means[valid_std] + stds[valid_std],
                color=optimizer["color"],
                alpha=0.15,
                linewidth=0,
            )

        plotted_anything = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric["ylabel"])
    ax.set_title(metric["title"])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=True)

    if not plotted_anything:
        print(
            f"WARNING: no data available for {metric_name}; "
            "no plot saved."
        )
        plt.close(fig)
        return

    # Register the save callback before displaying the preview.
    save_figure_on_close(fig, metric)

    if SHOW_PREVIEW:
        plt.show(block=WAIT_FOR_PREVIEW_CLOSE)
    else:
        # Without a preview window, there is no user close action.
        # Save directly so the figure is not silently discarded.
        if SAVE_SVG:
            svg_path = OUT_DIR / f"{metric['filename']}_{RUN_TIMESTAMP}.svg"
            fig.savefig(
                svg_path,
                format=SAVE_FORMAT,
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=PAD_INCHES,
            )
            print(f"Saved: {svg_path}")

        if SAVE_PNG:
            png_path = OUT_DIR / f"{metric['filename']}_{RUN_TIMESTAMP}.png"
            fig.savefig(
                png_path,
                format="png",
                dpi=SAVE_DPI,
                bbox_inches="tight",
                pad_inches=PAD_INCHES,
            )
            print(f"Saved: {png_path}")

    plt.close(fig)


# ============================================================
# EXPORT EPOCH-WISE CSV
# ============================================================

def export_epochwise_csv(data: dict):
    """Export individual epoch values and cross-seed statistics."""

    output_path = OUT_DIR / f"wikitext2_gpt_epochwise_metrics_{RUN_TIMESTAMP}.csv"

    columns = [
        "dataset",
        "model",
        "optimizer",
        "seed",
        "metric",
        "epoch",
        "value",
        "mean_across_seeds",
        "sample_std_across_seeds",
        "number_of_seeds",
    ]

    rows = []

    for optimizer in OPTIMIZERS:
        optimizer_name = optimizer["name"]

        for metric in METRICS:
            metric_name = metric["key"]

            stats_by_epoch = {
                item["epoch"]: item
                for item in epoch_statistics(
                    data,
                    optimizer_name,
                    metric_name,
                )
            }

            for seed in SEEDS:
                seed_data = (
                    data.get(optimizer_name, {})
                    .get(seed, {})
                    .get(metric_name, {})
                )

                for epoch, value in sorted(seed_data.items()):
                    stats = stats_by_epoch.get(epoch, {})

                    rows.append({
                        "dataset": "WikiText-2",
                        "model": MODEL_ARCH,
                        "optimizer": optimizer["label"],
                        "seed": seed,
                        "metric": metric_name,
                        "epoch": epoch,
                        "value": value,
                        "mean_across_seeds": stats.get("mean"),
                        "sample_std_across_seeds": stats.get("std"),
                        "number_of_seeds": stats.get("n_seeds"),
                    })

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


# ============================================================
# EXPORT FINAL-EPOCH SUMMARY
# ============================================================

def export_final_epochs_csv(data: dict):
    """
    Summarize the last N available epochs for each seed, then summarize
    those per-seed means across seeds.
    """

    output_path = OUT_DIR / f"wikitext2_gpt_final10_summary_{RUN_TIMESTAMP}.csv"

    columns = [
        "optimizer",
        "metric",
        "seed",
        "number_of_epochs",
        "mean_over_last_epochs",
        "std_over_last_epochs",
        "last_epoch_included",
        "cross_seed_mean_of_seed_means",
        "cross_seed_sample_std_of_seed_means",
        "number_of_seeds",
    ]

    rows = []

    for optimizer in OPTIMIZERS:
        optimizer_name = optimizer["name"]

        for metric in METRICS:
            metric_name = metric["key"]
            per_seed_means = []

            for seed in SEEDS:
                values_by_epoch = (
                    data.get(optimizer_name, {})
                    .get(seed, {})
                    .get(metric_name, {})
                )

                ordered = sorted(values_by_epoch.items())

                if not ordered:
                    continue

                selected = ordered[-LAST_N_EPOCHS:]
                values = np.asarray(
                    [value for _, value in selected],
                    dtype=float,
                )

                seed_mean = float(np.mean(values))
                seed_std = (
                    float(np.std(values, ddof=1))
                    if len(values) >= 2
                    else float("nan")
                )

                per_seed_means.append(seed_mean)

                rows.append({
                    "optimizer": optimizer["label"],
                    "metric": metric_name,
                    "seed": seed,
                    "number_of_epochs": len(selected),
                    "mean_over_last_epochs": seed_mean,
                    "std_over_last_epochs": seed_std,
                    "last_epoch_included": selected[-1][0],
                    "cross_seed_mean_of_seed_means": "",
                    "cross_seed_sample_std_of_seed_means": "",
                    "number_of_seeds": "",
                })

            if per_seed_means:
                rows.append({
                    "optimizer": optimizer["label"],
                    "metric": metric_name,
                    "seed": "MEAN_ACROSS_SEEDS",
                    "number_of_epochs": "",
                    "mean_over_last_epochs": "",
                    "std_over_last_epochs": "",
                    "last_epoch_included": "",
                    "cross_seed_mean_of_seed_means": float(
                        np.mean(per_seed_means)
                    ),
                    "cross_seed_sample_std_of_seed_means": (
                        float(np.std(per_seed_means, ddof=1))
                        if len(per_seed_means) >= 2
                        else float("nan")
                    ),
                    "number_of_seeds": len(per_seed_means),
                })

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


# ============================================================
# EXPORT COMPACT FINAL-10 TABLE CSV
# ============================================================

def export_final10_table_csv(data: dict):
    """
    Average epochs 41-50 within each seed, then calculate the mean and
    sample standard deviation across the three seed means.

    All ten epochs must exist for every seed before an aggregate is reported.
    """

    expected_epochs = list(range(41, 51))
    metric_names = [metric["key"] for metric in METRICS]

    output_path = OUT_DIR / f"wikitext2_gpt_table_final10_{RUN_TIMESTAMP}.csv"

    columns = ["optimizer"]

    for metric in metric_names:
        columns.extend([
            f"{metric}_seed_1993_last10_mean",
            f"{metric}_seed_1402_last10_mean",
            f"{metric}_seed_2025_last10_mean",
            f"{metric}_mean_across_seeds",
            f"{metric}_std_across_seeds",
            f"{metric}_epochs_used",
            f"{metric}_n_seeds",
        ])

    rows = []

    for optimizer in OPTIMIZERS:
        optimizer_name = optimizer["name"]
        row = {"optimizer": optimizer["label"]}

        for metric in metric_names:
            seed_means = []
            missing_seed_data = []

            for seed in SEEDS:
                seed_column = f"{metric}_seed_{seed}_last10_mean"

                epoch_values = (
                    data.get(optimizer_name, {})
                    .get(seed, {})
                    .get(metric, {})
                )

                missing_epochs = [
                    epoch
                    for epoch in expected_epochs
                    if epoch not in epoch_values
                ]

                if missing_epochs:
                    row[seed_column] = ""
                    missing_seed_data.append(
                        f"seed={seed}, missing_epochs={missing_epochs}"
                    )
                    continue

                final_values = np.asarray(
                    [epoch_values[epoch] for epoch in expected_epochs],
                    dtype=float,
                )

                seed_mean = float(np.mean(final_values))
                row[seed_column] = seed_mean
                seed_means.append(seed_mean)

            if len(seed_means) == len(SEEDS):
                row[f"{metric}_mean_across_seeds"] = float(
                    np.mean(seed_means)
                )
                row[f"{metric}_std_across_seeds"] = float(
                    np.std(seed_means, ddof=1)
                )
                row[f"{metric}_epochs_used"] = "41-50"
                row[f"{metric}_n_seeds"] = len(seed_means)
            else:
                row[f"{metric}_mean_across_seeds"] = ""
                row[f"{metric}_std_across_seeds"] = ""
                row[f"{metric}_epochs_used"] = ""
                row[f"{metric}_n_seeds"] = len(seed_means)

                print(
                    f"WARNING: incomplete final-10 data for "
                    f"{optimizer['label']} / {metric}: "
                    + "; ".join(missing_seed_data)
                )

        rows.append(row)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved final-10-epoch table CSV: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("WikiText-2 GPT: Fairseq log plotting")
    print("=" * 80)
    print(f"Project root: {REPO}")
    print(f"Checkpoint root: {CHECKPOINT_ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Seeds: {SEEDS}")
    print()

    if not CHECKPOINT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {CHECKPOINT_ROOT}"
        )

    data = collect_all_runs()

    # Export CSV files before opening plot windows.
    # This ensures CSV files are written even when plot previews block.
    print("\nExporting CSV files...")
    export_epochwise_csv(data)
    export_final_epochs_csv(data)
    export_final10_table_csv(data)

    print("\nGenerating plots...")
    for metric in METRICS:
        plot_metric(data, metric)

    print("\n" + "=" * 80)
    print("Finished.")
    print(f"Outputs are in: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()