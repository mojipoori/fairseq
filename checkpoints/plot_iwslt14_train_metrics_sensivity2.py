#!/usr/bin/env python3

"""
Extract Fairseq epoch metrics across multiple seeds and generate:

1. Four separate plots for the 3-seed mean ± STD.

2. Four separate plots for the single-seed (seed1993) runs.

3. One standalone legend containing all 12 conditions.

4. Two CSV files containing statistics for ALL VALIDATION METRICS:

   Validation metrics:
       - valid_loss
       - valid_bleu
       - valid_ppl

   A. Single-seed final-10-epoch statistics for seed1993:
      - Mean of the last 10 validation epochs
      - Sample STD across those same 10 epochs

      This STD measures variation across the final 10 epochs
      of seed1993 only.

   B. Three-seed final-10-epoch statistics:
      - First compute the mean of the last 10 validation epochs
        separately for each seed.
      - Then compute the mean across those seed-level means.
      - Then compute the sample STD across those seed-level means.

Three-seed mean plots:
    - 3 epsilon conditions for Emoji
    - 3 epsilon conditions for AdaBelief
    - Total: 6 curves

Single-seed plots:
    - All 6 epsilon conditions for Emoji using seed1993
    - All 6 epsilon conditions for AdaBelief using seed1993
    - Total: 12 curves

For the single-seed plots, seed1993 is extracted from all runs,
including the runs that also have three-seed experiments.

The figures themselves do not contain legends.

A separate standalone legend is generated.

Marker mapping:

    1e-8  -> circle
    1e-12 -> square
    1e-16 -> triangle
    1e-6  -> cross
    1e-10 -> diamond
    1e-24 -> star

Markers are hollow in both the plots and the standalone legend.

Single-seed linestyle mapping for the three additional epsilon values:

    1e-6  -> :
    1e-10 -> --
    1e-24 -> -.
"""


from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import t


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
# EPSILON VALUES
# ============================================================
SINGLE_EPSILONS = [
    1e-24,
    1e-16,
    1e-12,
    1e-10,
    1e-8,
    1e-6,
]

THREE_SEED_EPSILONS = [
    1e-16,
    1e-12,
    1e-8,
]


# ============================================================
# VALIDATION METRICS FOR CSV
# ============================================================
VALIDATION_METRICS = [
    "valid_loss",
    "valid_bleu",
    "valid_ppl",
]


# ============================================================
# MARKERS BY EPSILON
# ============================================================
MARKERS_BY_EPSILON = {
    1e-8: "o",    # Circle
    1e-12: "s",   # Square
    1e-16: "^",   # Triangle
    1e-6: "x",    # Cross
    1e-10: "D",   # Diamond
    1e-24: "*",   # Star
}


# ============================================================
# SINGLE-SEED LINESTYLES
# ============================================================
SINGLE_LINESTYLES_BY_EPSILON = {
    1e-6: ":",
    1e-10: "--",
    1e-24: "-.",
}


# ============================================================
# RUNS
# ============================================================
RUNS = [

    # --------------------------------------------------------
    # Emoji — 3 seeds
    # --------------------------------------------------------
    {
        "label": "Emoji",
        "color": "#0057B8",
        "folder": "iwslt14_emoji",
        "seeds": SEEDS,
        "show_ci": True,
        "linewidth": 1.0,
        "marker": MARKERS_BY_EPSILON[1e-8],
        "linestyle": "-",
    },

    {
        "label": r"Emoji ($\varepsilon=10^{-12}$)",
        "color": "#1677C8",
        "folder": "iwslt14_emoji_12",
        "seeds": SEEDS,
        "show_ci": True,
        "linewidth": 1.0,
        "marker": MARKERS_BY_EPSILON[1e-12],
        "linestyle": "-",
    },

    {
        "label": r"Emoji ($\varepsilon=10^{-16}$)",
        "color": "#3F91C9",
        "folder": "iwslt14_emoji_16",
        "seeds": SEEDS,
        "show_ci": True,
        "linewidth": 1.0,
        "marker": MARKERS_BY_EPSILON[1e-16],
        "linestyle": "-",
    },

    # --------------------------------------------------------
    # Emoji — seed1993 only
    # --------------------------------------------------------
    {
        "label": r"Emoji ($\varepsilon=10^{-6}$)",
        "color": "#004A95",
        "folder": "iwslt14_emoji_6",
        "seeds": ["seed1993"],
        "show_ci": False,
        "linewidth": 4.0 / 3.0,
        "marker": MARKERS_BY_EPSILON[1e-6],
        "linestyle": SINGLE_LINESTYLES_BY_EPSILON[1e-6],
    },

    {
        "label": r"Emoji ($\varepsilon=10^{-10}$)",
        "color": "#0B6FC2",
        "folder": "iwslt14_emoji_10",
        "seeds": ["seed1993"],
        "show_ci": False,
        "linewidth": 4.0 / 3.0,
        "marker": MARKERS_BY_EPSILON[1e-10],
        "linestyle": SINGLE_LINESTYLES_BY_EPSILON[1e-10],
    },

    {
        "label": r"Emoji ($\varepsilon=10^{-24}$)",
        "color": "#3C8AD0",
        "folder": "iwslt14_emoji_24",
        "seeds": ["seed1993"],
        "show_ci": False,
        "linewidth": 4.0 / 3.0,
        "marker": MARKERS_BY_EPSILON[1e-24],
        "linestyle": SINGLE_LINESTYLES_BY_EPSILON[1e-24],
    },

    # --------------------------------------------------------
    # AdaBelief — 3 seeds
    # --------------------------------------------------------
    {
        "label": "AdaBelief",
        "color": "#B2182B",
        "folder": "iwslt14_adabelief",
        "seeds": SEEDS,
        "show_ci": True,
        "linewidth": 1.0,
        "marker": MARKERS_BY_EPSILON[1e-8],
        "linestyle": "-",
    },

    {
        "label": r"AdaBelief ($\varepsilon=10^{-12}$)",
        "color": "#D43D4F",
        "folder": "iwslt14_adabelief_12",
        "seeds": SEEDS,
        "show_ci": True,
        "linewidth": 1.0,
        "marker": MARKERS_BY_EPSILON[1e-12],
        "linestyle": "-",
    },

    {
        "label": r"AdaBelief ($\varepsilon=10^{-16}$)",
        "color": "#E66A73",
        "folder": "iwslt14_adabelief_16",
        "seeds": SEEDS,
        "show_ci": True,
        "linewidth": 1.0,
        "marker": MARKERS_BY_EPSILON[1e-16],
        "linestyle": "-",
    },

    # --------------------------------------------------------
    # AdaBelief — seed1993 only
    # --------------------------------------------------------
    {
        "label": r"AdaBelief ($\varepsilon=10^{-6}$)",
        "color": "#8C1020",
        "folder": "iwslt14_adabelief_6",
        "seeds": ["seed1993"],
        "show_ci": False,
        "linewidth": 4.0 / 3.0,
        "marker": MARKERS_BY_EPSILON[1e-6],
        "linestyle": SINGLE_LINESTYLES_BY_EPSILON[1e-6],
    },

    {
        "label": r"AdaBelief ($\varepsilon=10^{-10}$)",
        "color": "#C62F3F",
        "folder": "iwslt14_adabelief_10",
        "seeds": ["seed1993"],
        "show_ci": False,
        "linewidth": 4.0 / 3.0,
        "marker": MARKERS_BY_EPSILON[1e-10],
        "linestyle": SINGLE_LINESTYLES_BY_EPSILON[1e-10],
    },

    {
        "label": r"AdaBelief ($\varepsilon=10^{-24}$)",
        "color": "#E54352",
        "folder": "iwslt14_adabelief_24",
        "seeds": ["seed1993"],
        "show_ci": False,
        "linewidth": 4.0 / 3.0,
        "marker": MARKERS_BY_EPSILON[1e-24],
        "linestyle": SINGLE_LINESTYLES_BY_EPSILON[1e-24],
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
FIGSIZE = (8, 5)

SAVE_DPI = 300

SHOW_PREVIEW = True

WAIT_FOR_PREVIEW_CLOSE = True

SAVE_FORMAT = "svg"


# ============================================================
# STANDALONE LEGEND SETTINGS
# ============================================================
LEGEND_FILENAME = (
    "iwslt14_epoch_metrics_legend.svg"
)

LEGEND_FIGSIZE = (10, 5.0)


# ============================================================
# CSV SETTINGS
# ============================================================
SINGLE_SEED_CSV_FILENAME = (
    "iwslt14_epsilon_sensitivity_single_seed_last10_stats.csv"
)

THREE_SEED_CSV_FILENAME = (
    "iwslt14_epsilon_sensitivity_three_seed_last10_stats.csv"
)


# ============================================================
# PLOT MARKER SETTINGS
# ============================================================
PLOT_MARKERSIZE = 5.5

LEGEND_MARKERSIZE = 8.0

MARKER_EDGEWIDTH = 1.0


# ============================================================
# PARSE LOG
# ============================================================
def parse_log(
    path: Path,
) -> dict[str, dict[int, float]]:

    """Parse Fairseq JSON log lines into per-epoch metric series."""

    series: dict[str, dict[int, float]] = {
        "train_loss": {},
        "valid_loss": {},
        "valid_bleu": {},
        "valid_ppl": {},
    }

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

            if epoch is None:
                continue

            epoch = int(epoch)

            for key in series:

                if key in row:

                    series[key][epoch] = float(
                        row[key]
                    )

    return series


# ============================================================
# COLLECT RUN DATA
# ============================================================
def collect_run_data(
    run: dict,
) -> dict[str, dict[str, dict[int, float]]]:

    """
    Collect metric data for the seeds assigned to this run.

    Returns:
        metric -> seed -> epoch -> value
    """

    data: dict[str, dict[str, dict[int, float]]] = {
        "train_loss": {},
        "valid_loss": {},
        "valid_bleu": {},
        "valid_ppl": {},
    }

    run_seeds = run.get(
        "seeds",
        SEEDS,
    )

    for seed in run_seeds:

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

        parsed = parse_log(
            log_path
        )

        for metric in data:

            if parsed.get(metric):

                data[metric][seed] = (
                    parsed[metric]
                )

    return data



# ============================================================
# MEAN ± SAMPLE STANDARD DEVIATION, THE FUNCTION def compute_mean_ci() COMPUTES STANDARD DEVIATION RATHER THAN CONFIDENCE INTERVAL 95%
# ============================================================
def compute_mean_ci(
    seed_series: dict[str, dict[int, float]],
    allow_single_seed: bool = False,
) -> tuple[
    list[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute epoch-wise mean ± sample standard deviation.

    For three-seed runs:
        mean = mean across available seeds at each epoch
        lower = mean - sample STD (ddof=1)
        upper = mean + sample STD (ddof=1)

    For a single-seed run:
        lower = upper = mean
        No across-seed variability can be estimated.
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

        if len(values) < 2:
            if not (
                allow_single_seed
                and len(seed_series) == 1
                and len(values) == 1
            ):
                continue

        values = np.asarray(values, dtype=float)
        n = len(values)
        mean = np.mean(values)

        if n >= 2:
            std = np.std(values, ddof=1)
        else:
            std = 0.0

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
# GET FINAL N VALUES
# ============================================================
def get_final_n_values(
    log_path: Path,
    metric: str,
    n_epochs: int = 10,
) -> tuple[list[int], np.ndarray]:

    """
    Extract the final n available epoch values for a metric.
    """

    parsed = parse_log(
        log_path
    )

    series = parsed.get(
        metric,
        {},
    )

    if not series:

        raise ValueError(
            f"No '{metric}' values found in log:\n"
            f"{log_path}"
        )

    epochs = sorted(
        series.keys()
    )

    if len(epochs) < n_epochs:

        raise ValueError(
            f"Only {len(epochs)} epochs found for "
            f"'{metric}', but {n_epochs} are required:\n"
            f"{log_path}"
        )

    final_epochs = epochs[-n_epochs:]

    values = np.asarray(
        [
            series[epoch]
            for epoch in final_epochs
        ],
        dtype=float,
    )

    return (
        final_epochs,
        values,
    )


# ============================================================
# GET FINAL N STATS
# ============================================================
def get_final_n_stats(
    log_path: Path,
    metric: str,
    n_epochs: int = 10,
) -> tuple[list[int], float, float]:

    """
    Compute the mean and sample STD across the final n epochs
    of one run.
    """

    epochs, values = get_final_n_values(
        log_path,
        metric,
        n_epochs,
    )

    mean = float(
        np.mean(values)
    )

    if len(values) >= 2:

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

    else:

        std = 0.0

    return (
        epochs,
        mean,
        std,
    )


# ============================================================
# GET FINAL MEAN OF VALID BLEU
# ============================================================
def get_final_mean_bleu(
    log_path: Path,
) -> float:

    """
    Return the mean validation BLEU over the final 10 epochs.
    """

    _, values = get_final_n_values(
        log_path,
        "valid_bleu",
        10,
    )

    return float(
        np.mean(values)
    )


# ============================================================
# GET RUN EPSILON
# ============================================================
def get_run_epsilon(
    run: dict,
) -> float:

    """
    Infer epsilon from the run folder.
    """

    folder = run["folder"]

    if folder in (
        "iwslt14_emoji",
        "iwslt14_adabelief",
    ):

        return 1e-8

    exponent = int(
        folder.rsplit("_", 1)[-1]
    )

    return 10 ** (-exponent)


# ============================================================
# GET RUN OPTIMIZER
# ============================================================
def get_run_optimizer(
    run: dict,
) -> str:

    if run["folder"].startswith(
        "iwslt14_emoji"
    ):

        return "Emoji"

    return "AdaBelief"


# ============================================================
# SAVE SINGLE-SEED CSV
# ============================================================
def save_single_seed_csv(
    out_path: Path,
) -> None:

    """
    Save final-10-epoch statistics for seed1993
    for ALL validation metrics.

    One row per:

        configuration × validation metric

    Columns:

        optimizer
        epsilon
        metric
        seed
        n_epochs
        epochs
        mean_last_10_epochs
        std_across_last_10_epochs
    """

    rows = []

    print()
    print("=" * 82)
    print(
        "SINGLE-SEED CSV — ALL VALIDATION METRICS"
    )
    print("=" * 82)

    for run in RUNS:

        optimizer = get_run_optimizer(
            run
        )

        epsilon = get_run_epsilon(
            run
        )

        log_path = (
            ROOT
            / SELECTED_SEED
            / run["folder"]
            / "train.log"
        )

        if not log_path.exists():

            print(
                f"WARNING: missing log for CSV:\n"
                f"{log_path}"
            )

            continue

        for metric in VALIDATION_METRICS:

            try:

                (
                    epochs,
                    mean,
                    std,
                ) = get_final_n_stats(
                    log_path=log_path,
                    metric=metric,
                    n_epochs=10,
                )

                rows.append(
                    {
                        "optimizer": optimizer,
                        "epsilon": f"{epsilon:.0e}",
                        "metric": metric,
                        "seed": SELECTED_SEED,
                        "n_epochs": len(epochs),
                        "epochs": ",".join(
                            str(epoch)
                            for epoch in epochs
                        ),
                        "mean_last_10_epochs": (
                            f"{mean:.10f}"
                        ),
                        "std_across_last_10_epochs": (
                            f"{std:.10f}"
                        ),
                    }
                )

                print(
                    f"{optimizer:10s} | "
                    f"epsilon={epsilon:.0e} | "
                    f"{metric:10s} | "
                    f"mean={mean:.6f} | "
                    f"STD(last 10)={std:.6f}"
                )

            except Exception as exc:

                print(
                    f"WARNING: failed to calculate "
                    f"{metric} for {log_path}: "
                    f"{exc}"
                )

    fieldnames = [
        "optimizer",
        "epsilon",
        "metric",
        "seed",
        "n_epochs",
        "epochs",
        "mean_last_10_epochs",
        "std_across_last_10_epochs",
    ]

    with out_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        f"Wrote {out_path}"
    )


# ============================================================
# SAVE THREE-SEED CSV
# ============================================================
def save_three_seed_csv(
    out_path: Path,
) -> None:

    """
    Save three-seed final-10-epoch statistics
    for ALL validation metrics.

    For each configuration and metric:

        1. Compute the mean of the final 10 epochs
           independently for each seed.

        2. Compute the mean across the resulting
           seed-level means.

        3. Compute the sample STD across those seed-level means.

    One row per:

        configuration × validation metric

    Columns:

        optimizer
        epsilon
        metric
        seed1993_mean_last_10
        seed1402_mean_last_10
        seed2025_mean_last_10
        n_valid_seeds
        mean_across_seeds
        std_across_seeds
    """

    rows = []

    print()
    print("=" * 82)
    print(
        "THREE-SEED CSV — ALL VALIDATION METRICS"
    )
    print("=" * 82)

    for run in RUNS:

        # ----------------------------------------------------
        # Only original three-seed runs.
        # ----------------------------------------------------
        if len(
            run.get(
                "seeds",
                SEEDS,
            )
        ) != 3:

            continue

        optimizer = get_run_optimizer(
            run
        )

        epsilon = get_run_epsilon(
            run
        )

        # ----------------------------------------------------
        # Process every validation metric separately.
        # ----------------------------------------------------
        for metric in VALIDATION_METRICS:

            seed_means: dict[str, float] = {}

            # ------------------------------------------------
            # Calculate final-10 mean independently
            # for every seed.
            # ------------------------------------------------
            for seed in SEEDS:

                log_path = (
                    ROOT
                    / seed
                    / run["folder"]
                    / "train.log"
                )

                if not log_path.exists():

                    print(
                        f"WARNING: missing log:\n"
                        f"{log_path}"
                    )

                    continue

                try:

                    (
                        _,
                        mean,
                        _,
                    ) = get_final_n_stats(
                        log_path=log_path,
                        metric=metric,
                        n_epochs=10,
                    )

                    seed_means[seed] = mean

                    print(
                        f"{optimizer:10s} | "
                        f"epsilon={epsilon:.0e} | "
                        f"{metric:10s} | "
                        f"{seed} | "
                        f"last10 mean={mean:.6f}"
                    )

                except Exception as exc:

                    print(
                        f"WARNING: failed to calculate "
                        f"{metric} for {log_path}: "
                        f"{exc}"
                    )

            # ------------------------------------------------
            # Need at least two seed-level means for
            # sample STD across seeds.
            # ------------------------------------------------
            if len(seed_means) < 2:

                print(
                    f"WARNING: {optimizer}, "
                    f"epsilon={epsilon:.0e}, "
                    f"metric={metric}: "
                    f"fewer than two valid seed means."
                )

                continue

            values = np.asarray(
                list(seed_means.values()),
                dtype=float,
            )

            mean_across_seeds = float(
                np.mean(values)
            )

            std_across_seeds = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            rows.append(
                {
                    "optimizer": optimizer,

                    "epsilon": f"{epsilon:.0e}",

                    "metric": metric,

                    "seed1993_mean_last_10": (
                        f"{seed_means['seed1993']:.10f}"
                        if "seed1993" in seed_means
                        else ""
                    ),

                    "seed1402_mean_last_10": (
                        f"{seed_means['seed1402']:.10f}"
                        if "seed1402" in seed_means
                        else ""
                    ),

                    "seed2025_mean_last_10": (
                        f"{seed_means['seed2025']:.10f}"
                        if "seed2025" in seed_means
                        else ""
                    ),

                    "n_valid_seeds": len(
                        seed_means
                    ),

                    "mean_across_seeds": (
                        f"{mean_across_seeds:.10f}"
                    ),

                    "std_across_seeds": (
                        f"{std_across_seeds:.10f}"
                    ),
                }
            )

            print(
                f"{optimizer:10s} | "
                f"epsilon={epsilon:.0e} | "
                f"{metric:10s} | "
                f"3-seed mean={mean_across_seeds:.6f} | "
                f"3-seed STD={std_across_seeds:.6f}"
            )

    fieldnames = [
        "optimizer",
        "epsilon",
        "metric",
        "seed1993_mean_last_10",
        "seed1402_mean_last_10",
        "seed2025_mean_last_10",
        "n_valid_seeds",
        "mean_across_seeds",
        "std_across_seeds",
    ]

    with out_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        f"Wrote {out_path}"
    )


# ============================================================
# PLOT METRIC
# ============================================================
def plot_metric(
    runs_data: list[tuple[dict, dict]],
    plot_cfg: dict,
    out_path: Path,
    plot_type: str,
) -> bool:

    """
    Create, preview, and save one metric plot.

    plot_type:

        "mean_std"
            Only the three-seed runs are plotted.
            Total = 6 curves.
            STD is shown.

        "single_seed"
            seed1993 is extracted from ALL runs.
            Total = 12 curves.
            No CI is shown.
    """

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
    )

    plotted = False

    # ========================================================
    # Z-ORDER
    # ========================================================
    zorders = {

        # Emoji
        "Emoji": 12,

        r"Emoji ($\varepsilon=10^{-6}$)": 11,

        r"Emoji ($\varepsilon=10^{-10}$)": 10,

        r"Emoji ($\varepsilon=10^{-12}$)": 9,

        r"Emoji ($\varepsilon=10^{-16}$)": 8,

        r"Emoji ($\varepsilon=10^{-24}$)": 7,

        # AdaBelief
        "AdaBelief": 6,

        r"AdaBelief ($\varepsilon=10^{-6}$)": 5,

        r"AdaBelief ($\varepsilon=10^{-10}$)": 4,

        r"AdaBelief ($\varepsilon=10^{-12}$)": 3,

        r"AdaBelief ($\varepsilon=10^{-16}$)": 2,

        r"AdaBelief ($\varepsilon=10^{-24}$)": 1,
    }

    # ========================================================
    # PLOT EACH RUN
    # ========================================================
    for run, data in runs_data:

        is_single_seed_run = (
            len(
                run.get(
                    "seeds",
                    SEEDS,
                )
            ) == 1
        )

        # ----------------------------------------------------
        # THREE-SEED MEAN PLOT
        # ----------------------------------------------------
        if plot_type == "mean_std":

            if is_single_seed_run:
                continue

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
            ) = compute_mean_ci(
                seed_series,
                allow_single_seed=False,
            )

        # ----------------------------------------------------
        # SINGLE-SEED PLOT
        # ----------------------------------------------------
        elif plot_type == "single_seed":

            # Take seed1993 from ALL runs.
            seed_series_all = data.get(
                plot_cfg["key"],
                {},
            )

            if SELECTED_SEED not in seed_series_all:
                continue

            seed_series = {
                SELECTED_SEED:
                    seed_series_all[SELECTED_SEED]
            }

            (
                epochs,
                means,
                lower,
                upper,
            ) = compute_mean_ci(
                seed_series,
                allow_single_seed=True,
            )

        else:

            raise ValueError(
                f"Unknown plot_type: {plot_type}"
            )

        # ----------------------------------------------------
        # CHECK DATA
        # ----------------------------------------------------
        if len(epochs) == 0:
            continue

        epochs = np.asarray(
            epochs
        )

        # ----------------------------------------------------
        # Run-specific style
        # ----------------------------------------------------
        line_width = run.get(
            "linewidth",
            2.0,
        )

        marker = run.get(
            "marker",
            "o",
        )

        linestyle = run.get(
            "linestyle",
            "-",
        )

        show_ci = run.get(
            "show_ci",
            True,
        )

        # ----------------------------------------------------
        # Plot line + hollow markers
        # ----------------------------------------------------
        ax.plot(
            epochs,
            means,
            color=run["color"],
            label=run["label"],
            linewidth=line_width,
            alpha=0.85,
            marker=marker,
            markersize=PLOT_MARKERSIZE,
            markevery=max(
                1,
                len(epochs) // 20,
            ),
            linestyle=linestyle,

            # Hollow markers
            markerfacecolor="none",
            markeredgecolor=run["color"],
            markeredgewidth=MARKER_EDGEWIDTH,

            zorder=zorders.get(
                run["label"],
                1,
            ),
        )

        # ----------------------------------------------------
        # STD
        #
        # Only for the three-seed mean plots.
        # ----------------------------------------------------
        if (
            plot_type == "mean_std"
            and show_ci
        ):

            ax.fill_between(
                epochs,
                lower,
                upper,
                color=run["color"],
                alpha=0.14,
                zorder=0,
            )

        plotted = True

    # ========================================================
    # CHECK
    # ========================================================
    if not plotted:

        plt.close(fig)

        return False

    # ========================================================
    # AXES
    # ========================================================
    ax.set_xlabel(
        "Epoch"
    )

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

    # ========================================================
    # NO LEGEND ON THE PLOT
    # ========================================================
    # Legend is generated separately.
    # ========================================================

    # ========================================================
    # LAYOUT
    # ========================================================
    fig.tight_layout()

    # ========================================================
    # PREVIEW
    # ========================================================
    if SHOW_PREVIEW:

        print()

        if plot_type == "mean_std":

            print(
                f"Preview: "
                f"{plot_cfg['title']} — "
                f"3-seed mean ± STD"
            )

        else:

            print(
                f"Preview: "
                f"{plot_cfg['title']} — "
                f"single seed {SELECTED_SEED}"
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
        bbox_inches="tight",
    )

    print(
        f"Wrote {out_path}"
    )

    plt.close(fig)

    return True


# ============================================================
# STANDALONE LEGEND
# ============================================================
def create_standalone_legend(
    out_path: Path,
) -> None:

    """
    Create the same normal standalone legend as before.

    All 12 conditions are included.

    Original labels are preserved.

    Marker shapes are applied according to epsilon.

    All markers are hollow.
    """

    fig = plt.figure(
        figsize=LEGEND_FIGSIZE,
    )

    handles = []
    labels = []

    # --------------------------------------------------------
    # Create one normal legend entry for every RUN.
    # --------------------------------------------------------
    for run in RUNS:

        handle = Line2D(
            [0],
            [0],

            color=run["color"],

            linewidth=run.get(
                "linewidth",
                2.0,
            ),

            linestyle=run.get(
                "linestyle",
                "-",
            ),

            marker=run.get(
                "marker",
                "o",
            ),

            markersize=LEGEND_MARKERSIZE,

            # Hollow marker
            markerfacecolor="none",

            # Colored outline
            markeredgecolor=run["color"],

            markeredgewidth=MARKER_EDGEWIDTH,

            alpha=0.85,
        )

        handles.append(
            handle
        )

        # Preserve original label exactly.
        labels.append(
            run["label"]
        )

    # --------------------------------------------------------
    # Original normal legend structure.
    # --------------------------------------------------------
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncols=2,
        frameon=True,
        fontsize=9.5,
        markerscale=0.9,
        handlelength=2.2,
        handleheight=0.8,
        handletextpad=0.6,
        labelspacing=0.55,
        borderpad=0.6,
        borderaxespad=0.4,
        columnspacing=1.4,
        numpoints=1,
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------
    if SHOW_PREVIEW:

        print()

        print(
            "Preview: Standalone epoch metrics legend"
        )

        print(
            "Close the preview window to continue."
        )

        plt.show(
            block=WAIT_FOR_PREVIEW_CLOSE
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    fig.savefig(
        out_path,
        format=SAVE_FORMAT,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    print(
        f"Wrote {out_path}"
    )

    plt.close(fig)


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

    print(
        f"  Seeds used: "
        f"{run.get('seeds', SEEDS)}"
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

    # ========================================================
    # LOAD ALL RUN DATA
    # ========================================================
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

    # ========================================================
    # THREE-SEED MEAN ± STD PLOTS
    # ========================================================
    print()

    print("=" * 82)

    print(
        "GENERATING THREE-SEED MEAN ± STD PLOTS"
    )

    print("=" * 82)

    for plot_cfg in PLOTS:

        out_path = (
            OUT_DIR
            / f"{plot_cfg['filename']}_mean_std.svg"
        )

        ok = plot_metric(
            runs_data=runs_data,
            plot_cfg=plot_cfg,
            out_path=out_path,
            plot_type="mean_std",
        )

        if not ok:

            print(
                f"Skipped "
                f"{plot_cfg['title']} "
                "mean ± CI plot "
                "(metric missing or "
                "insufficient seed data)"
            )

    # ========================================================
    # SINGLE-SEED PLOTS
    # ========================================================
    print()

    print("=" * 82)

    print(
        f"GENERATING SINGLE-SEED PLOTS — {SELECTED_SEED}"
    )

    print("=" * 82)

    for plot_cfg in PLOTS:

        out_path = (
            OUT_DIR
            / f"{plot_cfg['filename']}_single_seed.svg"
        )

        ok = plot_metric(
            runs_data=runs_data,
            plot_cfg=plot_cfg,
            out_path=out_path,
            plot_type="single_seed",
        )

        if not ok:

            print(
                f"Skipped "
                f"{plot_cfg['title']} "
                "single-seed plot "
                "(metric missing)"
            )

    # ========================================================
    # STANDALONE LEGEND
    # ========================================================
    print()

    print("=" * 82)

    print(
        "GENERATING STANDALONE LEGEND"
    )

    print("=" * 82)

    legend_out_path = (
        OUT_DIR
        / LEGEND_FILENAME
    )

    create_standalone_legend(
        legend_out_path
    )

    # ========================================================
    # SINGLE-SEED CSV
    # ========================================================
    print()

    print("=" * 82)

    print(
        "GENERATING SINGLE-SEED CSV — ALL VALIDATION METRICS"
    )

    print("=" * 82)

    single_seed_csv_path = (
        OUT_DIR
        / SINGLE_SEED_CSV_FILENAME
    )

    save_single_seed_csv(
        single_seed_csv_path
    )

    # ========================================================
    # THREE-SEED CSV
    # ========================================================
    print()

    print("=" * 82)

    print(
        "GENERATING THREE-SEED CSV — ALL VALIDATION METRICS"
    )

    print("=" * 82)

    three_seed_csv_path = (
        OUT_DIR
        / THREE_SEED_CSV_FILENAME
    )

    save_three_seed_csv(
        three_seed_csv_path
    )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()