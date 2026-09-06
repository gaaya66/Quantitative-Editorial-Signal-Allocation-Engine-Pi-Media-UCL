from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BACKTEST_RESULTS_PATH = Path("data/processed/backtest_results.csv")
ROBUSTNESS_RESULTS_PATH = Path("data/processed/robustness_results.csv")

PRECISION_FIGURE_PATH = Path(
    "reports/precision_at_5_comparison.png"
)

ROBUSTNESS_FIGURE_PATH = Path(
    "reports/robustness_momentum.png"
)


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_backtest_results(path: Path) -> pd.DataFrame:
    """Load and validate the main walk-forward backtest results."""

    df = pd.read_csv(path)

    required_columns = {
        "week",
        "next_week",
        "precision_momentum",
        "precision_baseline",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing backtest columns: {sorted(missing)}"
        )

    if df["week"].duplicated().any():
        raise ValueError(
            "Duplicate evaluation weeks found in backtest results."
        )

    precision_columns = [
        "precision_momentum",
        "precision_baseline",
    ]

    for column in precision_columns:
        if not df[column].between(0, 1).all():
            raise ValueError(
                f"Precision values outside [0, 1] found in {column}."
            )

    return df


def load_robustness_results(path: Path) -> pd.DataFrame:
    """Load and validate the momentum robustness analysis."""

    df = pd.read_csv(path)

    required_columns = {
        "short_window",
        "long_window",
        "n_periods",
        "mean_precision",
        "baseline_mean_precision",
        "baseline_n_periods",
        "lift_vs_baseline",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing robustness columns: {sorted(missing)}"
        )

    if len(df) != 3:
        raise ValueError(
            f"Expected 3 robustness specifications, found {len(df)}."
        )

    specification_columns = [
        "short_window",
        "long_window",
    ]

    if df[specification_columns].duplicated().any():
        raise ValueError(
            "Duplicate robustness specifications found."
        )

    if not df["mean_precision"].between(0, 1).all():
        raise ValueError(
            "Robustness mean Precision@5 values outside [0, 1]."
        )

    return df


# ---------------------------------------------------------------------
# Figure 1: Precision@5 comparison
# ---------------------------------------------------------------------

def plot_precision_comparison(df: pd.DataFrame) -> None:
    """Plot mean Precision@5 for momentum and the baseline."""

    methods = [
        "Baseline",
        "Momentum",
    ]

    values = [
        df["precision_baseline"].mean(),
        df["precision_momentum"].mean(),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(methods, values)

    ax.set_ylabel("Mean Precision@5")
    ax.set_title("Walk-Forward Prediction Performance")
    ax.set_ylim(0, 0.8)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    PRECISION_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        PRECISION_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: Momentum robustness
# ---------------------------------------------------------------------

def plot_robustness(df: pd.DataFrame) -> None:
    """
    Plot momentum robustness across different rolling-window choices.

    X-axis:
        Momentum specification.

    Y-axis:
        Mean Precision@5.

    The baseline is shown as a horizontal reference line.
    """

    df = df.sort_values(
        ["short_window", "long_window"]
    ).reset_index(drop=True)

    labels = [
        f"{int(row.short_window)}/{int(row.long_window)}"
        for row in df.itertuples()
    ]

    momentum_values = df["mean_precision"].tolist()
    baseline_value = df["baseline_mean_precision"].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(labels))

    bars = ax.bar(
        x,
        momentum_values,
    )

    ax.axhline(
        baseline_value,
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline ({baseline_value:.1%})",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)

    ax.set_xlabel("Momentum Window (Short / Long)")
    ax.set_ylabel("Mean Precision@5")
    ax.set_title("Momentum Robustness Across Window Specifications")
    ax.set_ylim(0, 0.8)

    for bar, value in zip(bars, momentum_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    ROBUSTNESS_FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        ROBUSTNESS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run all visualisation tasks."""

    backtest_results = load_backtest_results(
        BACKTEST_RESULTS_PATH
    )

    robustness_results = load_robustness_results(
        ROBUSTNESS_RESULTS_PATH
    )

    plot_precision_comparison(
        backtest_results
    )

    plot_robustness(
        robustness_results
    )

    print("Visualisation validation passed.")
    print(
        f"Wrote: {PRECISION_FIGURE_PATH}"
    )
    print(
        f"Wrote: {ROBUSTNESS_FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()