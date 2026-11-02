#!/usr/bin/env python3

"""Extract Fairseq epoch metrics across multiple seeds and plot mean ± 1 SD."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    # {
    #     "label": "AdaBelief (ε=1e-16)",
    #     "color": "#8C564B",
    #     "folder": "iwslt14_adabelief_16",
    # },
    # {
    #     "label": "AdaBelief (ε=1e-12)",
    #     "color": "#17BECF",
    #     "folder": "iwslt14_adabelief_12",
    # },
    # {
    #     "label": "AdamW",
    #     "color": "#9467BD",
    #     "folder": "iwslt14_adamw",
    # },
    # {
    #     "label": "Lion",
    #     "color": "#E377C2",
    #     "folder": "iwslt14_lion",
    # },
    {
        "label": "Emoji (α=0, Without Centering)",
        "color": "#FF7F0E",
        "folder": "iwslt14_emojiabl",
    },
    # {
    #     "label": "Emoji (ε=1e-12)",
    #     "color": "#2CA02C",
    #     "folder": "iwslt14_emoji_12",
    # },
    # {
    #     "label": "Emoji (ε=1e-16)",
    #     "color": "#7F7F7F",
    #     "folder": "iwslt14_emoji_16",
    # },
]


# ============================================================
# PLOT CONFIGURATION
# ============================================================

PLOTS = [
    {
        "key": "train_loss",
        "ylabel": "Train Loss",
        "title": "Train Loss vs Epoch",
        "filename": "train_loss_vs_epoch.svg",
    },
    {
        "key": "valid_loss",
        "ylabel": "Validation Loss",
        "title": "Validation Loss vs Epoch",
        "filename": "valid_loss_vs_epoch.svg",
    },
    {
        "key": "valid_bleu",
        "ylabel": "Validation BLEU",
        "title": "Validation BLEU vs Epoch",
        "filename": "valid_bleu_vs_epoch.svg",
    },
    {
        "key": "valid_ppl",
        "ylabel": "Validation PPL",
        "title": "Validation PPL vs Epoch",
        "filename": "valid_ppl_vs_epoch.svg",
    },
]


# ============================================================
# FIGURE / SAVE SETTINGS
# ============================================================

# Figure size in inches.
# Change these values whenever you want.
FIGSIZE = (8, 5)

# DPI used for saving raster/vector output.
SAVE_DPI = 300

# Preview before saving.
SHOW_PREVIEW = True

# Close the preview window before continuing to the next plot.
WAIT_FOR_PREVIEW_CLOSE = True

# Save format.
SAVE_FORMAT = "svg"


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
# MEAN + 1 SD
# ============================================================

def compute_mean_std(
    seed_series: dict[str, dict[int, float]],
) -> tuple[
    list[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute epoch-wise mean and sample standard deviation
    across seeds.

    The shaded band represents:

        Mean ± 1 SD
    """

    all_epochs = sorted(
        {
            epoch
            for series in seed_series.values()
            for epoch in series
        }
    )

    epochs = []
    means = []
    lower = []
    upper = []

    for epoch in all_epochs:

        values = [
            series[epoch]
            for series in seed_series.values()
            if epoch in series
        ]

        # Need at least two observations for sample SD.
        if len(values) < 2:
            continue

        values = np.asarray(
            values,
            dtype=float,
        )

        mean = np.mean(values)

        std = np.std(
            values,
            ddof=1,
        )

        epochs.append(epoch)
        means.append(mean)
        lower.append(mean - std)
        upper.append(mean + std)

    return (
        epochs,
        np.asarray(means),
        np.asarray(lower),
        np.asarray(upper),
    )


# ============================================================
# PLOT
# ============================================================

def plot_metric(
    runs_data: list[tuple[dict, dict]],
    plot_cfg: dict,
    out_path: Path,
) -> bool:
    """Create, preview, and save one metric plot."""

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
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

        seed_series = data.get(
            plot_cfg["key"],
            {},
        )

        if not seed_series:
            continue

        (
            epochs,
            means,
            lower,
            upper,
        ) = compute_mean_std(seed_series)

        if len(epochs) == 0:
            continue

        epochs = np.asarray(epochs)

        # ----------------------------------------------------
        # Mean line
        # ----------------------------------------------------

        ax.plot(
            epochs,
            means,
            color=run["color"],
            label=run["label"],
            linewidth=2.0,
            alpha=0.8,
            marker="o",
            markersize=3.5,
            markevery=max(
                1,
                len(epochs) // 20,
            ),
            zorder=zorders[run["label"]],
        )

        # ----------------------------------------------------
        # ± 1 Standard Deviation
        # ----------------------------------------------------

        ax.fill_between(
            epochs,
            lower,
            upper,
            color=run["color"],
            alpha=0.18,
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlabel("Epoch")

    ax.set_ylabel(
        plot_cfg["ylabel"]
    )

    ax.set_title(
        plot_cfg["title"]
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    ax.legend(
        frameon=True,
    )

    fig.tight_layout()

    # ========================================================
    # PREVIEW
    # ========================================================

    if SHOW_PREVIEW:

        print()
        print(
            f"Preview: {plot_cfg['title']}"
        )

        print(
            "Close the preview window "
            "to continue."
        )

        plt.show(
            block=WAIT_FOR_PREVIEW_CLOSE
        )

    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(
        out_path,
        format=SAVE_FORMAT,
        dpi=SAVE_DPI,
    )

    print(
        f"Wrote {out_path}"
    )

    plt.close(fig)

    return True


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    run: dict,
    data: dict,
) -> None:
    """Print number of seeds and epochs available."""

    print(
        f"{run['label']}:"
    )

    for metric in (
        "train_loss",
        "valid_loss",
        "valid_bleu",
        "valid_ppl",
    ):

        seed_data = data.get(
            metric,
            {},
        )

        print(
            f"  {metric}: "
            f"{len(seed_data)} seeds"
        )

        for seed, epochs in seed_data.items():

            print(
                f"    {seed}: "
                f"{len(epochs)} epochs"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_data = []

    # --------------------------------------------------------
    # Load all optimizer data
    # --------------------------------------------------------

    for run in RUNS:

        data = collect_run_data(
            run
        )

        print_summary(
            run,
            data,
        )

        runs_data.append(
            (
                run,
                data,
            )
        )

    print()

    if not runs_data:
        raise SystemExit(
            "No train.log files found; "
            "nothing to plot."
        )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    for plot_cfg in PLOTS:

        out_path = (
            OUT_DIR
            / plot_cfg["filename"]
        )

        ok = plot_metric(
            runs_data,
            plot_cfg,
            out_path,
        )

        if not ok:

            print(
                f"Skipped "
                f"{plot_cfg['title']} "
                "(metric missing or "
                "insufficient seed data)"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()