#!/usr/bin/env python3

"""Paired t-tests and SVG tables on the mean of the last 10 epochs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel


# ============================================================
# PATHS
# ============================================================

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "checkpoints_of_seeds"
OUT_DIR = ROOT / "plots"


# ============================================================
# SETTINGS
# ============================================================

SEEDS = [
    "seed1993",
    "seed1402",
    "seed2025",
]

METRIC = "valid_bleu"
LAST_N_EPOCHS = 10

ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

SHOW_PREVIEW = True
WAIT_FOR_PREVIEW_CLOSE = True

SAVE_FORMAT = "svg"
SAVE_DPI = 300


# ============================================================
# RUNS
# ============================================================

RUNS = {
    "Emoji": "iwslt14_emoji",
    "Emoji_1e-12": "iwslt14_emoji_12",
    "Emoji_1e-16": "iwslt14_emoji_16",

    "AdaBelief": "iwslt14_adabelief",
    "AdaBelief_1e-12": "iwslt14_adabelief_12",
    "AdaBelief_1e-16": "iwslt14_adabelief_16",

    "AdaBelief_v2": "iwslt14_adabeliefv2",
    "AdamW": "iwslt14_adamw",
    "Lion": "iwslt14_lion",
}



REFERENCE_OPTIMIZER = "Emoji"


COMPARISONS = [
    
    "AdaBelief",
    "AdaBelief_1e-12",
    "AdaBelief_1e-16",
]


# ============================================================
# BUILD PAIRS
# ============================================================

def build_pairs() -> list[tuple[str, str]]:
    """Build reference-vs-optimizer pairs."""

    if REFERENCE_OPTIMIZER not in RUNS:
        raise ValueError(
            f"Unknown reference optimizer: "
            f"{REFERENCE_OPTIMIZER}"
        )

    if not COMPARISONS:
        raise ValueError(
            "COMPARISONS cannot be empty."
        )

    unknown_optimizers = [
        optimizer
        for optimizer in COMPARISONS
        if optimizer not in RUNS
    ]

    if unknown_optimizers:
        raise ValueError(
            "Unknown optimizer(s) in COMPARISONS: "
            + ", ".join(unknown_optimizers)
        )

    if REFERENCE_OPTIMIZER in COMPARISONS:
        raise ValueError(
            f"Do not include '{REFERENCE_OPTIMIZER}' "
            "in COMPARISONS. "
            "The reference optimizer is added automatically."
        )

    if len(set(COMPARISONS)) != len(COMPARISONS):
        raise ValueError(
            "Duplicate optimizer found in COMPARISONS."
        )

    return [
        (REFERENCE_OPTIMIZER, optimizer)
        for optimizer in COMPARISONS
    ]


# ============================================================
# PARSE LOG
# ============================================================

def parse_log(path: Path) -> dict[int, float]:
    """Parse one Fairseq log and return metric values by epoch."""

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

            if epoch is None:
                continue

            if METRIC not in row:
                continue

            epoch = int(epoch)

            series[epoch] = float(
                row[METRIC]
            )

    return series


# ============================================================
# LAST N EPOCHS
# ============================================================

def mean_last_n_epochs(
    series: dict[int, float],
    n: int,
) -> tuple[list[int], float]:

    if len(series) < n:
        raise ValueError(
            f"Only {len(series)} epochs available, "
            f"but {n} are required."
        )

    epochs = sorted(series.keys())[-n:]

    values = np.asarray(
        [
            series[epoch]
            for epoch in epochs
        ],
        dtype=float,
    )

    return (
        epochs,
        float(
            np.mean(values)
        ),
    )


# ============================================================
# COLLECT SEED VALUES
# ============================================================

def collect_optimizer_values(
    optimizer: str,
) -> dict[str, float]:

    folder = RUNS[optimizer]

    results: dict[str, float] = {}

    for seed in SEEDS:

        log_path = (
            ROOT
            / seed
            / folder
            / "train.log"
        )

        if not log_path.exists():
            raise FileNotFoundError(
                f"Missing log:\n{log_path}"
            )

        series = parse_log(log_path)

        epochs, mean_value = (
            mean_last_n_epochs(
                series,
                LAST_N_EPOCHS,
            )
        )

        results[seed] = mean_value

        print(
            f"{optimizer:18s} | "
            f"{seed} | "
            f"epochs {epochs[0]}-{epochs[-1]} | "
            f"mean = {mean_value:.6f}"
        )

    return results


# ============================================================
# 95% CI FOR A MEAN
# ============================================================

def mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    if n < 2:
        raise ValueError(
            "At least two observations are required."
        )

    mean_value = np.mean(values)

    std_value = np.std(
        values,
        ddof=1,
    )

    standard_error = (
        std_value / np.sqrt(n)
    )

    critical_value = t.ppf(
        (1.0 + confidence) / 2.0,
        df=n - 1,
    )

    margin = (
        critical_value
        * standard_error
    )

    return (
        mean_value - margin,
        mean_value + margin,
    )


# ============================================================
# 95% CI FOR PAIRED DIFFERENCES
# ============================================================

def paired_ci(
    differences: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:

    differences = np.asarray(
        differences,
        dtype=float,
    )

    n = len(differences)

    if n < 2:
        raise ValueError(
            "At least two paired observations are required."
        )

    mean_difference = np.mean(
        differences
    )

    std_difference = np.std(
        differences,
        ddof=1,
    )

    standard_error = (
        std_difference / np.sqrt(n)
    )

    critical_value = t.ppf(
        (1.0 + confidence) / 2.0,
        df=n - 1,
    )

    margin = (
        critical_value
        * standard_error
    )

    return (
        mean_difference - margin,
        mean_difference + margin,
    )


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Holm step-down correction.

    The number of comparisons is determined automatically
    from the number of supplied p-values.

    Returns:
        adjusted_p_values
        reject
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p_values)

    if m == 0:
        raise ValueError(
            "At least one p-value is required."
        )

    order = np.argsort(
        p_values
    )

    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)

    for i, p_value in enumerate(
        sorted_p
    ):
        multiplier = m - i

        adjusted_sorted[i] = (
            p_value
            * multiplier
        )

    adjusted_sorted = (
        np.maximum.accumulate(
            adjusted_sorted
        )
    )

    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0,
    )

    adjusted = np.empty(m)

    adjusted[order] = (
        adjusted_sorted
    )

    reject = (
        adjusted < alpha
    )

    return adjusted, reject


# ============================================================
# PAIRED T-TEST
# ============================================================

def run_paired_test(
    values_a: dict[str, float],
    values_b: dict[str, float],
) -> dict:

    common_seeds = [
        seed
        for seed in SEEDS
        if seed in values_a
        and seed in values_b
    ]

    if len(common_seeds) < 2:
        raise ValueError(
            "At least two common seeds are required."
        )

    a = np.asarray(
        [
            values_a[seed]
            for seed in common_seeds
        ],
        dtype=float,
    )

    b = np.asarray(
        [
            values_b[seed]
            for seed in common_seeds
        ],
        dtype=float,
    )

    differences = a - b

    result = ttest_rel(
        a,
        b,
        alternative="two-sided",
    )

    mean_difference = np.mean(
        differences
    )

    ci_low, ci_high = paired_ci(
        differences,
        confidence=CONFIDENCE_LEVEL,
    )

    return {
        "seeds": common_seeds,
        "a_values": a,
        "b_values": b,
        "differences": differences,
        "mean_difference": mean_difference,
        "t": float(result.statistic),
        "df": int(result.df),
        "p": float(result.pvalue),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ============================================================
# TABLE 1 DATA
# ============================================================

def build_optimizer_summary(
    optimizer_values: dict[str, dict[str, float]],
) -> list[dict]:

    summary = []

    for optimizer in RUNS:

        seed_values = np.asarray(
            [
                optimizer_values[optimizer][seed]
                for seed in SEEDS
            ],
            dtype=float,
        )

        overall_mean = np.mean(
            seed_values
        )

        ci_low, ci_high = mean_ci(
            seed_values,
            confidence=CONFIDENCE_LEVEL,
        )

        summary.append(
            {
                "optimizer": optimizer,
                "seed1993": seed_values[0],
                "seed1402": seed_values[1],
                "seed2025": seed_values[2],
                "mean": overall_mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return summary


# ============================================================
# PANDAS DATAFRAME — TABLE 1
# ============================================================

def build_optimizer_dataframe(
    summary: list[dict],
) -> pd.DataFrame:

    rows = []

    for item in summary:

        rows.append(
            {
                "Optimizer": item["optimizer"],
                "Seed 1993": item["seed1993"],
                "Seed 1402": item["seed1402"],
                "Seed 2025": item["seed2025"],
                "Mean": item["mean"],
                "95% CI": (
                    f"[{item['ci_low']:.3f}, "
                    f"{item['ci_high']:.3f}]"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PANDAS DATAFRAME — TABLE 2
# ============================================================

def build_ttest_dataframe(
    results: list[dict],
) -> pd.DataFrame:

    rows = []

    for result in results:

        rows.append(
            {
                "Comparison": result["comparison"],
                "Mean Diff": result["mean_difference"],
                "t": result["t"],
                "df": result["df"],
                "Raw p": result["p"],
                "Holm-adjusted p": result["p_holm"],
                "95% CI": (
                    f"[{result['ci_low']:.3f}, "
                    f"{result['ci_high']:.3f}]"
                ),
                "Significant": (
                    "Yes"
                    if result["significant"]
                    else "No"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PRINT TABLE 1 IN TERMINAL
# ============================================================

def print_optimizer_summary_dataframe(
    df: pd.DataFrame,
) -> None:

    display_df = df.copy()

    display_df["Seed 1993"] = display_df[
        "Seed 1993"
    ].map(
        lambda x: f"{x:.3f}"
    )

    display_df["Seed 1402"] = display_df[
        "Seed 1402"
    ].map(
        lambda x: f"{x:.3f}"
    )

    display_df["Seed 2025"] = display_df[
        "Seed 2025"
    ].map(
        lambda x: f"{x:.3f}"
    )

    display_df["Mean"] = display_df[
        "Mean"
    ].map(
        lambda x: f"{x:.3f}"
    )

    print()

    print("=" * 72)
    print(
        "DATAFRAME — OPTIMIZER SUMMARY"
    )
    print("=" * 72)

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# PRINT TABLE 2 IN TERMINAL
# ============================================================

def print_ttest_dataframe(
    df: pd.DataFrame,
) -> None:

    display_df = df.copy()

    display_df["Mean Diff"] = display_df[
        "Mean Diff"
    ].map(
        lambda x: f"{x:+.6f}"
    )

    display_df["t"] = display_df[
        "t"
    ].map(
        lambda x: f"{x:.6f}"
    )

    display_df["Raw p"] = display_df[
        "Raw p"
    ].map(
        lambda x: f"{x:.6f}"
    )

    display_df["Holm-adjusted p"] = (
        display_df[
            "Holm-adjusted p"
        ].map(
            lambda x: f"{x:.6f}"
        )
    )

    print()

    print("=" * 72)
    print(
        "DATAFRAME — PAIRED T-TEST"
    )
    print("=" * 72)

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# TABLE 1 — MEAN + 95% CI
# ============================================================

def create_optimizer_summary_table(
    summary: list[dict],
    out_path: Path,
) -> None:

    columns = [
        "Optimizer",
        "Seed 1993\nMean BLEU",
        "Seed 1402\nMean BLEU",
        "Seed 2025\nMean BLEU",
        "Mean",
        "95% CI",
    ]

    rows = []

    for item in summary:

        rows.append(
            [
                item["optimizer"],
                f"{item['seed1993']:.3f}",
                f"{item['seed1402']:.3f}",
                f"{item['seed2025']:.3f}",
                f"{item['mean']:.3f}",
                (
                    f"[{item['ci_low']:.3f}, "
                    f"{item['ci_high']:.3f}]"
                ),
            ]
        )

    fig, ax = plt.subplots(
        figsize=(11, 3.8)
    )

    ax.axis("off")

    ax.set_title(
        "Validation BLEU — Mean of Last 10 Epochs per Seed",
        fontsize=13,
        pad=16,
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[
            0.20,
            0.16,
            0.16,
            0.16,
            0.12,
            0.20,
        ],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for cell in table.get_celld().values():
        cell.set_height(0.16)

    for cell in table.get_celld().values():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)

    for col in range(len(columns)):

        header_cell = table[(0, col)]

        header_cell.set_text_props(
            weight="bold"
        )

        header_cell.set_height(0.20)

    fig.tight_layout()

    if SHOW_PREVIEW:

        print()
        print(
            "Preview: Optimizer Mean + 95% CI Table"
        )
        print(
            "Close the preview window to continue."
        )

        plt.show(
            block=WAIT_FOR_PREVIEW_CLOSE
        )

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
# TABLE 2 — T-TEST
# ============================================================

def create_ttest_table(
    results: list[dict],
    out_path: Path,
) -> None:

    columns = [
        "Comparison",
        "Mean Diff",
        "t",
        "df",
        "Raw p",
        "Holm-adjusted p",
        "95% CI",
        "Significant",
    ]

    rows = []

    for result in results:

        rows.append(
            [
                result["comparison"],
                f"{result['mean_difference']:+.3f}",
                f"{result['t']:.3f}",
                f"{result['df']}",
                f"{result['p']:.6f}",
                f"{result['p_holm']:.6f}",
                (
                    f"[{result['ci_low']:.3f}, "
                    f"{result['ci_high']:.3f}]"
                ),
                (
                    "YES"
                    if result["significant"]
                    else "NO"
                ),
            ]
        )

    fig, ax = plt.subplots(
        figsize=(13, 3.2)
    )

    ax.axis("off")

    ax.set_title(
        "Paired t-Test Results with Holm Correction",
        fontsize=13,
        pad=16,
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[
            0.20,
            0.11,
            0.09,
            0.06,
            0.13,
            0.16,
            0.16,
            0.09,
        ],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for cell in table.get_celld().values():
        cell.set_height(0.18)

    for cell in table.get_celld().values():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)

    for col in range(len(columns)):

        header_cell = table[(0, col)]

        header_cell.set_text_props(
            weight="bold"
        )

        header_cell.set_height(0.22)

    fig.tight_layout()

    if SHOW_PREVIEW:

        print()
        print(
            "Preview: Paired t-Test Table"
        )
        print(
            "Close the preview window to continue."
        )

        plt.show(
            block=WAIT_FOR_PREVIEW_CLOSE
        )

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
# MAIN
# ============================================================

def main() -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Build selected reference-vs-optimizer pairs.
    PAIRS = build_pairs()

    print("=" * 72)

    print(
        f"Metric: {METRIC}"
    )

    print(
        f"Last {LAST_N_EPOCHS} epochs per seed"
    )

    print("=" * 72)

    print()

    print(
        f"Reference optimizer: {REFERENCE_OPTIMIZER}"
    )

    print()

    print(
        "Selected comparisons:"
    )

    for optimizer_a, optimizer_b in PAIRS:

        print(
            f"  {optimizer_a} vs {optimizer_b}"
        )

    print()

    print(
        f"Number of Holm comparisons: {len(PAIRS)}"
    )

    print()

    # --------------------------------------------------------
    # Collect one value per seed for each optimizer
    # --------------------------------------------------------

    optimizer_values = {}

    for optimizer in RUNS:

        print(
            f"\n{optimizer}"
        )

        print(
            "-" * 72
        )

        optimizer_values[optimizer] = (
            collect_optimizer_values(
                optimizer
            )
        )

    # --------------------------------------------------------
    # Optimizer summary
    # --------------------------------------------------------

    summary = build_optimizer_summary(
        optimizer_values
    )

    print()

    print("=" * 72)

    print(
        "OPTIMIZER MEAN + 95% CI"
    )

    print("=" * 72)

    for item in summary:

        print(
            f"{item['optimizer']:18s} | "
            f"Mean = {item['mean']:.6f} | "
            f"95% CI = "
            f"[{item['ci_low']:.6f}, "
            f"{item['ci_high']:.6f}]"
        )

    # --------------------------------------------------------
    # Build and print optimizer DataFrame
    # --------------------------------------------------------

    optimizer_df = build_optimizer_dataframe(
        summary
    )

    print_optimizer_summary_dataframe(
        optimizer_df
    )

    # --------------------------------------------------------
    # Paired t-tests
    # --------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "PAIRED T-TESTS"
    )

    print("=" * 72)

    results = []

    for optimizer_a, optimizer_b in PAIRS:

        result = run_paired_test(
            optimizer_values[optimizer_a],
            optimizer_values[optimizer_b],
        )

        result["comparison"] = (
            f"{optimizer_a} vs {optimizer_b}"
        )

        results.append(result)

    # --------------------------------------------------------
    # Holm correction
    # --------------------------------------------------------

    raw_p_values = [
        result["p"]
        for result in results
    ]

    corrected_p_values, reject = (
        holm_correction(
            raw_p_values,
            alpha=ALPHA,
        )
    )

    for i, result in enumerate(results):

        result["p_holm"] = (
            corrected_p_values[i]
        )

        result["significant"] = bool(
            reject[i]
        )

    # --------------------------------------------------------
    # Build and print paired t-test DataFrame
    # --------------------------------------------------------

    ttest_df = build_ttest_dataframe(
        results
    )

    print_ttest_dataframe(
        ttest_df
    )

    # --------------------------------------------------------
    # Detailed results
    # --------------------------------------------------------

    print()

    print(
        f"Holm correction uses "
        f"m = {len(results)} comparisons."
    )

    print()

    for result in results:

        print("-" * 72)

        print(
            result["comparison"]
        )

        print("-" * 72)

        print(
            "Seeds:"
        )

        for i, seed in enumerate(
            result["seeds"]
        ):

            print(
                f"  {seed}: "
                f"{result['a_values'][i]:.6f} "
                f"vs "
                f"{result['b_values'][i]:.6f} "
                f"-> "
                f"diff = "
                f"{result['differences'][i]:+.6f}"
            )

        print()

        print(
            f"Mean difference: "
            f"{result['mean_difference']:+.6f}"
        )

        print(
            f"t = "
            f"{result['t']:.6f}"
        )

        print(
            f"df = "
            f"{result['df']}"
        )

        print(
            f"p-value = "
            f"{result['p']:.6f}"
        )

        print(
            f"95% CI = "
            f"["
            f"{result['ci_low']:.6f}, "
            f"{result['ci_high']:.6f}"
            f"]"
        )

        print(
            f"Holm-adjusted p = "
            f"{result['p_holm']:.6f}"
        )

        print(
            "Significant after Holm: "
            f"{'YES' if result['significant'] else 'NO'}"
        )

    # --------------------------------------------------------
    # Generate SVG tables
    # --------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "GENERATING SVG TABLES"
    )

    print("=" * 72)

    summary_path = (
        OUT_DIR
        / "valid_bleu_mean_ci_table.svg"
    )

    ttest_path = (
        OUT_DIR
        / "valid_bleu_paired_ttest_table.svg"
    )

    create_optimizer_summary_table(
        summary,
        summary_path,
    )

    create_ttest_table(
        results,
        ttest_path,
    )

    print()

    print("=" * 72)

    print(
        "DONE"
    )

    print("=" * 72)

    print()

    print(
        "Output directory:"
    )

    print(
        OUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()