#!/usr/bin/env python3
"""
Extract Fairseq epoch metrics across multiple seeds and plot mean ± 1 SD.

Outputs:
    1. Epoch-wise plots with mean ± 1 SD across seeds.
    2. Final-10-epoch mean ± sample STD across seeds.
    3. Final-10-epoch mean for every individual seed.
    4. Epoch-wise mean ± sample STD across all optimizers.
    5. Epoch-wise raw metric values for every optimizer and every seed,
       arranged side-by-side in one comparative CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
# NUMBER OF FINAL EPOCHS FOR CSV SUMMARY
# ============================================================

LAST_N_EPOCHS = 10


# ============================================================
# FINAL CSV FILENAMES
# ============================================================

FINAL_CSV_FILENAME = (
    "iwslt14_all_metrics_mean_std_last10.csv"
)

PER_SEED_CSV_FILENAME = (
    "iwslt14_all_metrics_last10_per_seed.csv"
)

EPOCHWISE_CSV_FILENAME = (
    "iwslt14_all_metrics_epochwise_mean_std_comparative.csv"
)

SEEDWISE_CSV_FILENAME = (
    "iwslt14_all_metrics_epochwise_seed_comparative.csv"
)


# ============================================================
# RUNS
# ============================================================

RUNS = [
    {
        "label": "Emoji (Proposed)",
        "color": "#1F77B4",
        "folder": "iwslt14_emoji",
    },
    {
        "label": "AdaBelief",
        "color": "#D62728",
        "folder": "iwslt14_adabelief",
    },
    {
        "label": "AdaBelief (ε=1e-16)",
        "color": "#8C564B",
        "folder": "iwslt14_adabelief_16",
    },
    {
        "label": "AdaBelief (ε=1e-12)",
        "color": "#17BECF",
        "folder": "iwslt14_adabelief_12",
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
    {
        "label": "Emoji (ε=1e-12)",
        "color": "#2CA02C",
        "folder": "iwslt14_emoji_12",
    },
    {
        "label": "Emoji (ε=1e-16)",
        "color": "#7F7F7F",
        "folder": "iwslt14_emoji_16",
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

FIGSIZE = (8, 5)
SAVE_DPI = 300
SHOW_PREVIEW = True
WAIT_FOR_PREVIEW_CLOSE = True
SAVE_FORMAT = "svg"


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
# EPOCH-WISE MEAN + 1 SD
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
        std = np.std(values, ddof=1)

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
# EPOCH-WISE COMPARATIVE CSV
# ============================================================

def compute_epochwise_comparative_rows(
    runs_data: list[tuple[dict, dict]],
) -> list[dict]:
    """
    Compute epoch-wise mean and sample STD across seeds for every
    metric and all configured optimizers.

    One CSV row is created for each metric × epoch combination,
    with the optimizers placed side-by-side for comparison.
    """

    metric_keys = (
        "train_loss",
        "valid_loss",
        "valid_bleu",
        "valid_ppl",
    )

    prefixes = {
        "Emoji (Proposed)": "Emoji",
        "AdaBelief": "AdaBelief",
        "AdamW": "AdamW",
        "Lion": "Lion",
    }

    rows = []

    for metric in metric_keys:

        all_epochs = sorted(
            {
                epoch
                for run, data in runs_data
                for seed_series in [data.get(metric, {})]
                for series in seed_series.values()
                for epoch in series
            }
        )

        for epoch in all_epochs:

            row = {
                "model": "IWSLT14 Transformer",
                "task": "Machine Translation",
                "dataset": "IWSLT14 De-En",
                "metric": metric,
                "epoch": epoch,
            }

            for run, data in runs_data:

                label = run["label"]

                if label not in prefixes:
                    continue

                prefix = prefixes[label]
                seed_series = data.get(metric, {})

                values = [
                    series[epoch]
                    for series in seed_series.values()
                    if epoch in series
                ]

                if len(values) >= 2:

                    values = np.asarray(
                        values,
                        dtype=float,
                    )

                    row[f"{prefix}_mean"] = float(
                        np.mean(values)
                    )

                    row[f"{prefix}_std"] = float(
                        np.std(values, ddof=1)
                    )

                else:

                    row[f"{prefix}_mean"] = np.nan
                    row[f"{prefix}_std"] = np.nan

            rows.append(row)

    return rows


def save_epochwise_comparative_csv(
    runs_data: list[tuple[dict, dict]],
) -> None:
    """
    Save one comparative CSV containing epoch-wise mean and sample STD
    for all configured optimizers.
    """

    rows = compute_epochwise_comparative_rows(
        runs_data
    )

    if not rows:
        print(
            "\nNo epoch-wise comparative rows were generated."
        )
        return

    df = pd.DataFrame(rows)

    desired_columns = [
        "model",
        "task",
        "dataset",
        "metric",
        "epoch",
        "Emoji_mean",
        "Emoji_std",
        "AdaBelief_mean",
        "AdaBelief_std",
        "AdamW_mean",
        "AdamW_std",
        "Lion_mean",
        "Lion_std",
    ]

    df = df.reindex(columns=desired_columns)

    df = df.sort_values(
        by=["metric", "epoch"]
    ).reset_index(drop=True)

    output_path = (
        OUT_DIR
        / EPOCHWISE_CSV_FILENAME
    )

    df.to_csv(
        output_path,
        index=False,
        float_format="%.6f",
    )

    print()
    print("=" * 100)
    print(
        "EPOCH-WISE MEAN ± STD — COMPARATIVE CSV"
    )
    print("=" * 100)

    print(
        df.to_string(index=False)
    )

    print()
    print(
        "Saved epoch-wise comparative CSV:"
    )
    print(output_path)
    print("=" * 100)


# ============================================================
# EPOCH-WISE RAW SEED COMPARISON
# ============================================================

def compute_epochwise_seed_comparative_rows(
    runs_data: list[tuple[dict, dict]],
) -> list[dict]:
    """
    Create one comparative CSV containing raw epoch-wise values
    for every optimizer and every seed.

    Each row represents:
        metric × epoch

    Columns contain the corresponding value for each:
        seed × optimizer

    Example:
        seed1993_Emoji
        seed1993_AdaBelief
        seed1993_AdamW
        seed1993_Lion
        seed1402_Emoji
        ...
    """

    metric_keys = (
        "train_loss",
        "valid_loss",
        "valid_bleu",
        "valid_ppl",
    )

    optimizer_prefixes = {
        "Emoji (Proposed)": "Emoji",
        "AdaBelief": "AdaBelief",
        "AdamW": "AdamW",
        "Lion": "Lion",
    }

    # --------------------------------------------------------
    # Build lookup:
    #
    # lookup[(seed, optimizer, metric, epoch)] = value
    # --------------------------------------------------------

    lookup = {}

    for run, data in runs_data:

        optimizer_label = run["label"]

        if optimizer_label not in optimizer_prefixes:
            continue

        optimizer_prefix = optimizer_prefixes[
            optimizer_label
        ]

        for metric in metric_keys:

            seed_data = data.get(
                metric,
                {},
            )

            for seed, epoch_values in seed_data.items():

                for epoch, value in epoch_values.items():

                    lookup[
                        (
                            seed,
                            optimizer_prefix,
                            metric,
                            epoch,
                        )
                    ] = value

    # --------------------------------------------------------
    # Collect all metric × epoch combinations
    # --------------------------------------------------------

    metric_epochs = sorted(
        {
            (metric, epoch)
            for (
                seed,
                optimizer,
                metric,
                epoch,
            ) in lookup
        },
        key=lambda x: (
            metric_keys.index(x[0]),
            x[1],
        ),
    )

    rows = []

    for metric, epoch in metric_epochs:

        row = {
            "model": "IWSLT14 Transformer",
            "task": "Machine Translation",
            "dataset": "IWSLT14 De-En",
            "metric": metric,
            "epoch": epoch,
        }

        # ----------------------------------------------------
        # Keep seed order exactly as defined in SEEDS.
        # Keep optimizer order exactly as defined in RUNS.
        # ----------------------------------------------------

        for seed in SEEDS:

            for run in RUNS:

                optimizer_label = run["label"]

                if optimizer_label not in optimizer_prefixes:
                    continue

                optimizer_prefix = optimizer_prefixes[
                    optimizer_label
                ]

                column_name = (
                    f"{seed}_{optimizer_prefix}"
                )

                row[column_name] = lookup.get(
                    (
                        seed,
                        optimizer_prefix,
                        metric,
                        epoch,
                    ),
                    np.nan,
                )

        rows.append(row)

    return rows


def save_epochwise_seed_comparative_csv(
    runs_data: list[tuple[dict, dict]],
) -> None:
    """
    Save one CSV containing raw epoch-wise values for every
    optimizer and every seed.

    This is separate from the mean ± STD CSV and does not
    average across seeds.
    """

    rows = compute_epochwise_seed_comparative_rows(
        runs_data
    )

    if not rows:
        print(
            "\nNo epoch-wise seed-comparative rows were generated."
        )
        return

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Explicit column order
    # --------------------------------------------------------

    base_columns = [
        "model",
        "task",
        "dataset",
        "metric",
        "epoch",
    ]

    optimizer_prefixes = {
        "Emoji (Proposed)": "Emoji",
        "AdaBelief": "AdaBelief",
        "AdamW": "AdamW",
        "Lion": "Lion",
    }

    seed_optimizer_columns = []

    for seed in SEEDS:

        for run in RUNS:

            optimizer_label = run["label"]

            if optimizer_label not in optimizer_prefixes:
                continue

            optimizer_prefix = optimizer_prefixes[
                optimizer_label
            ]

            seed_optimizer_columns.append(
                f"{seed}_{optimizer_prefix}"
            )

    desired_columns = (
        base_columns
        + seed_optimizer_columns
    )

    df = df.reindex(
        columns=desired_columns
    )

    df = df.sort_values(
        by=["metric", "epoch"]
    ).reset_index(drop=True)

    output_path = (
        OUT_DIR
        / SEEDWISE_CSV_FILENAME
    )

    df.to_csv(
        output_path,
        index=False,
        float_format="%.6f",
    )

    print()
    print("=" * 100)
    print(
        "EPOCH-WISE RAW VALUES — ALL SEEDS × ALL OPTIMIZERS"
    )
    print("=" * 100)

    print(
        df.to_string(index=False)
    )

    print()
    print(
        "Saved seed-wise comparative CSV:"
    )
    print(output_path)
    print("=" * 100)


# ============================================================
# FINAL 10-EPOCH SEED MEAN
# ============================================================

def compute_final_seed_mean(
    seed_series: dict[int, float],
) -> float:
    """
    Compute the mean of the final LAST_N_EPOCHS
    available epochs for one seed.
    """

    if len(seed_series) < LAST_N_EPOCHS:

        raise ValueError(
            f"Only {len(seed_series)} epochs available, "
            f"but {LAST_N_EPOCHS} are required."
        )

    sorted_epochs = sorted(
        seed_series.keys()
    )

    final_epochs = sorted_epochs[
        -LAST_N_EPOCHS:
    ]

    final_values = [
        seed_series[epoch]
        for epoch in final_epochs
    ]

    return float(
        np.mean(
            np.asarray(
                final_values,
                dtype=float,
            )
        )
    )


# ============================================================
# FINAL MEAN ± STD ACROSS SEEDS
# ============================================================

def compute_final_summary(
    run: dict,
    data: dict,
) -> list[dict]:
    """
    For each metric:

    1. Compute the mean of the final 10 epochs
       within each seed.
    2. Compute mean and sample STD across those
       seed-level means.
    """

    rows = []

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

        if not seed_data:

            print(
                f"  WARNING: no data for "
                f"{run['label']} / {metric}"
            )

            continue

        seed_means = {}

        for seed, epoch_values in seed_data.items():

            try:

                seed_mean = compute_final_seed_mean(
                    epoch_values
                )

            except ValueError as exc:

                print(
                    f"  WARNING: {run['label']} / "
                    f"{metric} / {seed}: {exc}"
                )

                continue

            seed_means[seed] = seed_mean

        # Sample STD requires at least two seeds.

        if len(seed_means) < 2:

            print(
                f"  WARNING: {run['label']} / "
                f"{metric}: fewer than 2 valid seeds. "
                f"CSV row skipped."
            )

            continue

        seed_values = np.asarray(
            list(seed_means.values()),
            dtype=float,
        )

        mean = float(
            np.mean(seed_values)
        )

        std = float(
            np.std(
                seed_values,
                ddof=1,
            )
        )

        mean_std_text = (
            f"{mean:.4f} ± {std:.4f}"
        )

        rows.append(
            {
                "model": "IWSLT14 Transformer",
                "task": "Machine Translation",
                "dataset": "IWSLT14 De-En",
                "metric": metric,
                "optimizer": run["label"],
                "n_seeds": len(seed_means),
                "mean ± std": mean_std_text,
            }
        )

        print(
            f"  Final {LAST_N_EPOCHS} epochs | "
            f"{run['label']} | "
            f"{metric} | "
            f"seed means: "
            f"{', '.join(f'{k}={v:.4f}' for k, v in seed_means.items())} | "
            f"mean ± std = {mean_std_text}"
        )

    return rows


# ============================================================
# PER-SEED FINAL 10-EPOCH MEANS
# ============================================================

def compute_per_seed_last10_summary(
    run: dict,
    data: dict,
) -> list[dict]:
    """
    Compute the mean of the final LAST_N_EPOCHS separately
    for every seed and every metric.

    No averaging across seeds is performed here.
    """

    rows = []

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

        if not seed_data:

            print(
                f"  WARNING: no data for "
                f"{run['label']} / {metric}"
            )

            continue

        for seed, epoch_values in seed_data.items():

            try:

                seed_mean = compute_final_seed_mean(
                    epoch_values
                )

            except ValueError as exc:

                print(
                    f"  WARNING: {run['label']} / "
                    f"{metric} / {seed}: {exc}"
                )

                continue

            rows.append(
                {
                    "model": "IWSLT14 Transformer",
                    "task": "Machine Translation",
                    "dataset": "IWSLT14 De-En",
                    "metric": metric,
                    "optimizer": run["label"],
                    "seed": seed,
                    "last_10_epoch_mean": seed_mean,
                }
            )

    return rows


# ============================================================
# SAVE ONE CSV FOR ALL METRICS
# ============================================================

def save_final_summary_csv(
    rows: list[dict],
) -> None:
    """
    Save all metrics and all optimizers into one CSV file.

    This CSV contains:
        mean ± sample STD across seeds.
    """

    if not rows:

        print(
            "\nNo final summary rows were generated."
        )

        return

    df = pd.DataFrame(rows)

    output_path = (
        OUT_DIR
        / FINAL_CSV_FILENAME
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "FINAL MEAN ± STD SUMMARY — LAST 10 EPOCHS PER SEED"
    )
    print("=" * 100)

    print(
        df.to_string(index=False)
    )

    print()
    print(
        "Saved CSV:"
    )
    print(output_path)
    print("=" * 100)


# ============================================================
# SAVE PER-SEED CSV
# ============================================================

def save_per_seed_last10_csv(
    rows: list[dict],
) -> None:
    """
    Save the final-10-epoch mean separately for every:
        optimizer × metric × seed
    """

    if not rows:

        print(
            "\nNo per-seed final-10-epoch rows were generated."
        )

        return

    df = pd.DataFrame(rows)

    # Optional deterministic ordering.

    df = df.sort_values(
        by=[
            "metric",
            "optimizer",
            "seed",
        ]
    ).reset_index(drop=True)

    output_path = (
        OUT_DIR
        / PER_SEED_CSV_FILENAME
    )

    df.to_csv(
        output_path,
        index=False,
        float_format="%.6f",
    )

    print()
    print("=" * 100)
    print(
        "PER-SEED FINAL 10-EPOCH MEANS"
    )
    print("=" * 100)

    print(
        df.to_string(index=False)
    )

    print()
    print(
        "Saved per-seed CSV:"
    )
    print(output_path)
    print("=" * 100)


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
        "Emoji (Proposed)": 9,
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
        ) = compute_mean_std(
            seed_series
        )

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
            zorder=zorders[
                run["label"]
            ],
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

    # CSV 1:
    # mean ± STD across seeds

    final_csv_rows = []

    # CSV 2:
    # final 10-epoch mean for every individual seed

    per_seed_last10_rows = []

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

        # ----------------------------------------------------
        # 1. Final mean ± STD summary
        # ----------------------------------------------------

        final_rows = compute_final_summary(
            run,
            data,
        )

        final_csv_rows.extend(
            final_rows
        )

        # ----------------------------------------------------
        # 2. Per-seed final 10-epoch means
        # ----------------------------------------------------

        per_seed_rows = compute_per_seed_last10_summary(
            run,
            data,
        )

        per_seed_last10_rows.extend(
            per_seed_rows
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

    # for plot_cfg in PLOTS:

    #     out_path = (
    #         OUT_DIR
    #         / plot_cfg["filename"]
    #     )

    #     ok = plot_metric(
    #         runs_data,
    #         plot_cfg,
    #         out_path,
    #     )

    #     if not ok:

    #         print(
    #             f"Skipped "
    #             f"{plot_cfg['title']} "
    #             "(metric missing or "
    #             "insufficient seed data)"
    #         )

    # --------------------------------------------------------
    # Save ONE CSV containing:
    # mean ± STD across seeds
    # --------------------------------------------------------

    save_final_summary_csv(
        final_csv_rows
    )

    # --------------------------------------------------------
    # Save SECOND CSV containing:
    # final 10-epoch mean for each individual seed
    # --------------------------------------------------------

    save_per_seed_last10_csv(
        per_seed_last10_rows
    )

    # --------------------------------------------------------
    # Save THIRD CSV containing:
    # epoch-wise mean ± STD for all optimizers
    # --------------------------------------------------------

    save_epochwise_comparative_csv(
        runs_data
    )

    # --------------------------------------------------------
    # Save FOURTH CSV containing:
    # raw epoch-wise values for every seed × optimizer
    # --------------------------------------------------------

    save_epochwise_seed_comparative_csv(
        runs_data
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()