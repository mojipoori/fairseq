#!/usr/bin/env python3

"""
IWSLT14 epsilon sensitivity analysis for Emoji and AdaBelief.

Three plots are generated:

1. Single-seed plot:

   The selected single seed is plotted for all epsilon values.

2. Three-seed mean +/- sample STD plot:

   Three-seed mean and sample STD are shown for the epsilon values
   available for all three seeds.

3. Combined plot:

   Both the three-seed mean markers and the selected single-seed
   curves are shown together.

The combined plot shows ONLY the three-seed mean markers.
The three-seed STD band and connecting mean line are NOT shown
in the combined plot.

The single-seed curve uses the same optimizer hue as the corresponding
mean curve, but a substantially darker shade so that it remains visible.

The epsilon values shown are:

    1e-24, 1e-16, 1e-12, 1e-10, 1e-8, 1e-6

The three-seed statistics follow the supplied three-seed script:

the final 10 validation BLEU epochs are averaged within each seed,

then the mean and sample STD are computed across seeds.
"""

from __future__ import annotations

import csv
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

SELECTED_SEED = "seed1993"


# ============================================================
# NUMBER OF FINAL EPOCHS
# ============================================================
LAST_N_EPOCHS = 10


# ============================================================
# EPSILON VALUES
# ============================================================

# Single-seed curve: all available epsilon values.
SINGLE_EPSILONS = [
    1e-24,
    1e-16,
    1e-12,
    1e-10,
    1e-8,
    1e-6,
]

# Three-seed mean +/- STD: only the epsilon values for which
# the three-seed experiments are available.
THREE_SEED_EPSILONS = [
    1e-16,
    1e-12,
    1e-8,
]

# ============================================================
# METRICS
# ============================================================
METRICS = {
    "bleu": {
        "log_key": "valid_bleu",
        "label": "Validation BLEU",
        "slug": "valid_bleu",
    },
    "ppl": {
        "log_key": "valid_ppl",
        "label": "Validation Perplexity (PPL)",
        "slug": "valid_ppl",
    },
    "loss": {
        "log_key": "valid_loss",
        "label": "Validation Loss",
        "slug": "valid_loss",
    },
}

CURRENT_METRIC = "ppl"
METRIC_KEY = METRICS[CURRENT_METRIC]["log_key"]
METRIC_LABEL = METRICS[CURRENT_METRIC]["label"]
# ============================================================
# FOLDER HELPER
# ============================================================
def epsilon_folder(
    optimizer: str,
    epsilon: float,
) -> str:
    """
    Match the existing folder naming convention.

    Base epsilon (1e-8):
        iwslt14_emoji
        iwslt14_adabelief

    Other epsilons:
        iwslt14_emoji_<exponent>
        iwslt14_adabelief_<exponent>
    """
    prefix = (
        "iwslt14_emoji"
        if optimizer == "Emoji"
        else "iwslt14_adabelief"
    )

    exponent = int(round(np.log10(epsilon)))

    if exponent == -8:
        return prefix

    return f"{prefix}_{abs(exponent)}"


# ============================================================
# OPTIMIZER COLORS
# ============================================================

# Main curve: bright/saturated color.
# Single-seed curve: same hue, much darker shade.
COLORS = {
    "Emoji": {
        "mean": "#0057B8",
        "single": "#0B6FC2",
    },
    "AdaBelief": {
        "mean": "#C62F3F",
        "single": "#D43D4F",
    },
}


# ============================================================
# PLOT CONFIGURATION
# ============================================================
FIGSIZE = (8.5, 5.2)

SAVE_DPI = 300

SHOW_PREVIEW = True

WAIT_FOR_PREVIEW_CLOSE = True

SAVE_FORMAT = "svg"

# Existing combined plot filename.
OUTPUT_FILENAME = (
    "iwslt14_epsilon_sensitivity_valid_bleu_combined.svg"
)

# New single-seed plot filename.
SINGLE_OUTPUT_FILENAME = (
    "iwslt14_epsilon_sensitivity_valid_bleu_single_seed.svg"
)

# New three-seed mean +/- STD plot filename.
THREE_SEED_OUTPUT_FILENAME = (
    "iwslt14_epsilon_sensitivity_valid_bleu_mean_std.svg"
)

# ============================================================
# CSV OUTPUT FILENAMES
# ============================================================

# CSV for all six epsilon values of the selected single seed.
SINGLE_CSV_FILENAME = (
    "iwslt14_epsilon_sensitivity_valid_bleu_6_epsilons.csv"
)

# CSV for three-seed mean and sample STD.
THREE_SEED_CSV_FILENAME = (
    "iwslt14_epsilon_sensitivity_valid_bleu_3_epsilons_mean_std.csv"
)
# ============================================================
# PARSE LOG
# ============================================================
def parse_log(
    path: Path,
) -> dict[int, float]:

    series: dict[int, float] = {}

    with path.open(
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            line = line.strip()

            if not line.startswith("{"):
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = row.get("epoch")

            if epoch is None or METRIC_KEY not in row:
                continue

            try:
                epoch = int(epoch)
                value = float(row[METRIC_KEY])
            except (TypeError, ValueError):
                continue

            if np.isfinite(value):
                series[epoch] = value

    return series


# ============================================================
# GET MEAN OF FINAL N EPOCHS
# ============================================================
def get_final_mean_metric(
    log_path: Path,
) -> float:

    series = parse_log(log_path)

    if not series:
        raise ValueError(
            f"No valid {METRIC_KEY} values found in log:\n"
            f"{log_path}"
        )

    epochs = sorted(series.keys())

    if len(epochs) < LAST_N_EPOCHS:
        raise ValueError(
            f"Only {len(epochs)} validation epochs found, "
            f"but {LAST_N_EPOCHS} are required:\n"
            f"{log_path}"
        )

    final_epochs = epochs[-LAST_N_EPOCHS:]

    final_values = [
        series[epoch]
        for epoch in final_epochs
    ]

    return float(np.mean(final_values))


# ============================================================
# READ SINGLE-SEED RESULT
# ============================================================
def collect_single_seed_results(
    optimizer: str,
) -> dict[float, float]:

    results: dict[float, float] = {}

    for epsilon in SINGLE_EPSILONS:

        folder = epsilon_folder(
            optimizer,
            epsilon,
        )

        log_path = (
            ROOT
            / SELECTED_SEED
            / folder
            / "train.log"
        )

        print(
            f"{optimizer:10s} | "
            f"single | "
            f"epsilon={epsilon:.0e} | ",
            end="",
        )

        if not log_path.exists():
            raise FileNotFoundError(
                f"Missing log file:\n{log_path}"
            )

        value = get_final_mean_metric(log_path)

        results[epsilon] = value

        print(
f"{METRIC_KEY}={value:.6f}"        )

    return results


# ============================================================
# READ THREE-SEED RESULT
# ============================================================
def collect_seed_values(
    optimizer: str,
    epsilon: float,
) -> list[float]:

    folder = epsilon_folder(
        optimizer,
        epsilon,
    )

    values: list[float] = []

    for seed in SEEDS:

        log_path = (
            ROOT
            / seed
            / folder
            / "train.log"
        )

        if not log_path.exists():
            print(
                f"WARNING: missing log:\n{log_path}"
            )
            continue

        try:
            value = get_final_mean_metric(log_path)
            values.append(value)

        except Exception as exc:
            print(
                f"WARNING: failed to read {log_path}: "
                f"{exc}"
            )

    return values


def collect_three_seed_results(
    optimizer: str,
) -> dict[float, tuple[float, float]]:

    results: dict[float, tuple[float, float]] = {}

    for epsilon in THREE_SEED_EPSILONS:

        values = collect_seed_values(
            optimizer,
            epsilon,
        )

        if len(values) != len(SEEDS):
            raise ValueError(
                f"{optimizer}, epsilon={epsilon:.0e}: "
                f"expected {len(SEEDS)} valid seeds, "
                f"found {len(values)}."
            )

        values_array = np.asarray(
            values,
            dtype=float,
        )

        mean = float(
            np.mean(values_array)
        )

        std = float(
            np.std(
                values_array,
                ddof=1,
            )
        )

        results[epsilon] = (
            mean,
            std,
        )

        print(
            f"{optimizer:10s} | "
            f"3-seed  | "
            f"epsilon={epsilon:.0e} | "
            f"mean={mean:.6f} | "
            f"STD={std:.6f}"
        )

    return results


# ============================================================
# CSV OUTPUT: SIX EPSILON VALUES (SINGLE SEED)
# ============================================================
# ============================================================
# CSV OUTPUT: SIX EPSILON VALUES (SINGLE SEED)
# ============================================================
def export_single_seed_csv(
    single_results: dict[str, dict[float, float]],
    out_path: Path,
) -> None:

    metric_column = f"final_10_epoch_mean_{METRIC_KEY}"

    with out_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "optimizer",
                "seed",
                "epsilon",
                metric_column,
            ],
        )

        writer.writeheader()

        for optimizer in ["Emoji", "AdaBelief"]:
            for epsilon in SINGLE_EPSILONS:
                writer.writerow({
                    "optimizer": optimizer,
                    "seed": SELECTED_SEED,
                    "epsilon": f"{epsilon:.0e}",
                    metric_column: single_results[optimizer][epsilon],
                })

    print(f"Wrote CSV: {out_path}")


# ============================================================
# CSV OUTPUT: THREE EPSILON VALUES (THREE-SEED STATISTICS)
# ============================================================
def export_three_seed_csv(
    three_seed_results: dict[
        str,
        dict[float, tuple[float, float]],
    ],
    out_path: Path,
) -> None:

    mean_column = f"mean_{METRIC_KEY}"
    std_column = f"sample_std_{METRIC_KEY}"

    with out_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "optimizer",
                "epsilon",
                "number_of_valid_seeds",
                mean_column,
                std_column,
            ],
        )

        writer.writeheader()

        for optimizer in ["Emoji", "AdaBelief"]:
            for epsilon in THREE_SEED_EPSILONS:
                mean, std = three_seed_results[optimizer][epsilon]

                writer.writerow({
                    "optimizer": optimizer,
                    "epsilon": f"{epsilon:.0e}",
                    "number_of_valid_seeds": len(SEEDS),
                    mean_column: f"{mean:.10f}",
                    std_column: f"{std:.10f}",
                })

    print(f"Wrote CSV: {out_path}")

# ============================================================
# COMMON X AXIS CONFIGURATION
# ============================================================
def configure_x_axis(
    ax,
    single_x: np.ndarray,
) -> None:

    ax.set_xscale("log")

    ax.set_xticks(single_x)

    ax.set_xticklabels(
        [
            r"$10^{-24}$",
            r"$10^{-16}$",
            r"$10^{-12}$",
            r"$10^{-10}$",
            r"$10^{-8}$",
            r"$10^{-6}$",
        ]
    )

    ax.set_xlabel(
        r"$\varepsilon$",
        fontsize=11,
    )


# ============================================================
# COMMON Y AXIS CONFIGURATION
# ============================================================
def configure_y_axis(
    ax,
) -> None:

    ax.set_ylabel(
        f"Mean {METRIC_LABEL} over the Last 10 Epochs",
        fontsize=11,
    )


# ============================================================
# COMMON GRID CONFIGURATION
# ============================================================
def configure_grid(
    ax,
) -> None:

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.5,
        alpha=0.14,
    )


# ============================================================
# COMMON LIMIT CONFIGURATION
# ============================================================
def configure_limits(
    ax,
) -> None:

    ax.set_xlim(
        3e-25,
        3e-6,
    )


# ============================================================
# PLOT: SINGLE-SEED ONLY
# ============================================================
def plot_single_seed(
    single_results: dict[str, dict[float, float]],
    out_path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
    )

    single_x = np.asarray(
        SINGLE_EPSILONS,
        dtype=float,
    )

    for optimizer in [
        "Emoji",
        "AdaBelief",
    ]:

        colors = COLORS[optimizer]

        single = np.asarray(
            [
                single_results[optimizer][epsilon]
                for epsilon in SINGLE_EPSILONS
            ],
            dtype=float,
        )

        ax.plot(
            single_x,
            single,
            color=colors["single"],
            linewidth=1.2,
            linestyle=":",
            marker="o",
            markersize=2.2,
            markerfacecolor=colors["single"],
            markeredgecolor=colors["single"],
            markeredgewidth=0.4,
            label=f"{optimizer} single seed",
            zorder=10,
        )

    # ========================================================
    # X AXIS
    # ========================================================
    configure_x_axis(
        ax,
        single_x,
    )

    # ========================================================
    # Y AXIS
    # ========================================================
    configure_y_axis(ax)

    # ========================================================
    # GRID
    # ========================================================
    configure_grid(ax)

    # ========================================================
    # LIMITS
    # ========================================================
    configure_limits(ax)

    # ========================================================
    # LEGEND
    # ========================================================
    ax.legend(
        frameon=True,
        loc="best",
        fontsize=9.5,
    )

    fig.tight_layout()

    # ========================================================
    # PREVIEW
    # ========================================================
    if SHOW_PREVIEW:

        print()

        print(
            "Preview: IWSLT14 single-seed epsilon sensitivity"
        )

        print(
            "Close the preview window to continue."
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
        bbox_inches="tight",
    )

    print()

    print(
        f"Wrote: {out_path}"
    )

    plt.close(fig)


# ============================================================
# PLOT: THREE-SEED MEAN +/- STD ONLY
# ============================================================
def plot_three_seed_mean_std(
    three_seed_results: dict[
        str,
        dict[float, tuple[float, float]],
    ],
    out_path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
    )

    three_seed_x = np.asarray(
        THREE_SEED_EPSILONS,
        dtype=float,
    )

    for optimizer in [
        "Emoji",
        "AdaBelief",
    ]:

        colors = COLORS[optimizer]

        mean = np.asarray(
            [
                three_seed_results[optimizer][epsilon][0]
                for epsilon in THREE_SEED_EPSILONS
            ],
            dtype=float,
        )

        std = np.asarray(
            [
                three_seed_results[optimizer][epsilon][1]
                for epsilon in THREE_SEED_EPSILONS
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # STD band
        # ----------------------------------------------------
        ax.fill_between(
            three_seed_x,
            mean - std,
            mean + std,
            color=colors["mean"],
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )

        # ----------------------------------------------------
        # Three-seed mean
        # ----------------------------------------------------
        ax.plot(
            three_seed_x,
            mean,
            color=colors["mean"],
            linewidth=2,
            linestyle="-",
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=colors["mean"],
            markeredgewidth=1.3,
            label=f"{optimizer} mean $\\pm$ STD",
            zorder=5,
        )

    # ========================================================
    # X AXIS
    # ========================================================
    ax.set_xscale("log")

    ax.set_xticks(
        three_seed_x
    )

    ax.set_xticklabels(
        [
            r"$10^{-16}$",
            r"$10^{-12}$",
            r"$10^{-8}$",
        ]
    )

    ax.set_xlabel(
        r"$\varepsilon$",
        fontsize=11,
    )

    # ========================================================
    # Y AXIS
    # ========================================================
    configure_y_axis(ax)

    # ========================================================
    # GRID
    # ========================================================
    configure_grid(ax)

    # ========================================================
    # LIMITS
    # ========================================================
    configure_limits(ax)

    # ========================================================
    # LEGEND
    # ========================================================
    ax.legend(
        frameon=True,
        loc="best",
        fontsize=9.5,
    )

    fig.tight_layout()

    # ========================================================
    # PREVIEW
    # ========================================================
    if SHOW_PREVIEW:

        print()

        print(
            "Preview: IWSLT14 three-seed mean +/- STD epsilon sensitivity"
        )

        print(
            "Close the preview window to continue."
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
        bbox_inches="tight",
    )

    print()

    print(
        f"Wrote: {out_path}"
    )

    plt.close(fig)


# ============================================================
# PLOT: COMBINED
# ============================================================
def plot_combined(
    single_results: dict[str, dict[float, float]],
    three_seed_results: dict[
        str,
        dict[float, tuple[float, float]],
    ],
    out_path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
    )

    single_x = np.asarray(
        SINGLE_EPSILONS,
        dtype=float,
    )

    three_seed_x = np.asarray(
        THREE_SEED_EPSILONS,
        dtype=float,
    )

    for optimizer in [
        "Emoji",
        "AdaBelief",
    ]:

        colors = COLORS[optimizer]

        mean = np.asarray(
            [
                three_seed_results[optimizer][epsilon][0]
                for epsilon in THREE_SEED_EPSILONS
            ],
            dtype=float,
        )

        single = np.asarray(
            [
                single_results[optimizer][epsilon]
                for epsilon in SINGLE_EPSILONS
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # Three-seed mean markers ONLY
        #
        # No STD band.
        # No connecting mean line.
        # ----------------------------------------------------
        ax.plot(
            three_seed_x,
            mean,
            color=colors["mean"],
            linewidth=0,
            linestyle="None",
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=colors["mean"],
            markeredgewidth=1.3,
            label=f"{optimizer} mean",
            zorder=5,
        )

        # ----------------------------------------------------
        # Single-seed curve
        # ----------------------------------------------------
        ax.plot(
            single_x,
            single,
            color=colors["single"],
            linewidth=1.2,
            linestyle=":",
            marker="o",
            markersize=2.2,
            markerfacecolor=colors["single"],
            markeredgecolor=colors["single"],
            markeredgewidth=0.4,
            label=f"{optimizer} single seed",
            zorder=10,
        )

    # ========================================================
    # X AXIS
    # ========================================================
    ax.set_xscale("log")

    ax.set_xticks(single_x)

    ax.set_xticklabels(
        [
            r"$10^{-24}$",
            r"$10^{-16}$",
            r"$10^{-12}$",
            r"$10^{-10}$",
            r"$10^{-8}$",
            r"$10^{-6}$",
        ]
    )

    ax.set_xlabel(
        r"$\varepsilon$",
        fontsize=11,
    )

    # ========================================================
    # Y AXIS
    # ========================================================
    configure_y_axis(ax)

    # ========================================================
    # GRID
    # ========================================================
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.5,
        alpha=0.14,
    )

    # ========================================================
    # LIMITS
    # ========================================================
    ax.set_xlim(
        3e-25,
        3e-6,
    )

    # ========================================================
    # LEGEND
    # ========================================================
    ax.legend(
        frameon=True,
        loc="best",
        fontsize=9.5,
    )

    fig.tight_layout()

    # ========================================================
    # PREVIEW
    # ========================================================
    if SHOW_PREVIEW:

        print()

        print(
            "Preview: IWSLT14 combined epsilon sensitivity"
        )

        print(
            "Close the preview window to continue."
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
        bbox_inches="tight",
    )

    print()

    print(
        f"Wrote: {out_path}"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
# ============================================================
# MAIN
# ============================================================
def main() -> None:

    global CURRENT_METRIC
    global METRIC_KEY
    global METRIC_LABEL

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric_name in ["bleu", "ppl", "loss"]:

        CURRENT_METRIC = metric_name
        METRIC_KEY = METRICS[metric_name]["log_key"]
        METRIC_LABEL = METRICS[metric_name]["label"]
        metric_slug = METRICS[metric_name]["slug"]

        print()
        print("=" * 82)
        print(f"IWSLT14 EPSILON SENSITIVITY — {METRIC_LABEL}")
        print("=" * 82)

        print(f"Single seed: {SELECTED_SEED}")
        print(f"Three-seed analysis: {', '.join(SEEDS)}")
        print(f"Final validation epochs averaged: {LAST_N_EPOCHS}")

        single_results = {}
        three_seed_results = {}

        for optimizer in ["Emoji", "AdaBelief"]:

            print()
            print("-" * 82)
            print(f"{optimizer} — single seed")
            print("-" * 82)

            single_results[optimizer] = (
                collect_single_seed_results(optimizer)
            )

            print()
            print("-" * 82)
            print(f"{optimizer} — three seeds")
            print("-" * 82)

            three_seed_results[optimizer] = (
                collect_three_seed_results(optimizer)
            )

        # ----------------------------------------------------
        # CSV EXPORTS
        # ----------------------------------------------------
        export_single_seed_csv(
            single_results,
            OUT_DIR / SINGLE_CSV_FILENAME.replace(
                "valid_bleu", metric_slug
            ),
        )

        export_three_seed_csv(
            three_seed_results,
            OUT_DIR / THREE_SEED_CSV_FILENAME.replace(
                "valid_bleu", metric_slug
            ),
        )

        # ----------------------------------------------------
        # SINGLE-SEED PLOT
        # ----------------------------------------------------
        plot_single_seed(
            single_results,
            OUT_DIR / SINGLE_OUTPUT_FILENAME.replace(
                "valid_bleu", metric_slug
            ),
        )

        # ----------------------------------------------------
        # THREE-SEED MEAN +/- STD PLOT
        # ----------------------------------------------------
        plot_three_seed_mean_std(
            three_seed_results,
            OUT_DIR / THREE_SEED_OUTPUT_FILENAME.replace(
                "valid_bleu", metric_slug
            ),
        )

        # ----------------------------------------------------
        # COMBINED PLOT
        # ----------------------------------------------------
        plot_combined(
            single_results,
            three_seed_results,
            OUT_DIR / OUTPUT_FILENAME.replace(
                "valid_bleu", metric_slug
            ),
        )

        print()
        print(f"Completed: {METRIC_LABEL}")


if __name__ == "__main__":
    main()