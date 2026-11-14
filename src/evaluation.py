"""
evaluation.py
-------------
Phase 5b: summarise the walk-forward backtest.

Per method: mean / median / std Precision@5, number of periods, lift vs
baseline. Added: paired difference vs baseline with a block-bootstrap 95%
CI (weeks are autocorrelated, so blocks), and the random-selection level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BACKTEST_RESULTS_PATH = "data/processed/backtest_results.csv"
EVALUATION_RESULTS_PATH = "reports/evaluation_summary.csv"

METHOD_NAMES = ["momentum", "baseline"]
TOP_N = 5
N_TOPICS = 16
RANDOM_PRECISION = TOP_N / N_TOPICS  # 0.3125


def block_bootstrap_ci(x, block: int = 8, n_boot: int = 2000, seed: int = 0):
    """95% CI for the mean of an autocorrelated weekly series."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    block = min(block, len(x))
    n_blocks = int(np.ceil(len(x) / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, len(x) - block + 1, n_blocks)
        means[b] = np.concatenate([x[s:s + block] for s in starts])[:len(x)].mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def load_backtest_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["week", "next_week"])


def summarise_method(df: pd.DataFrame, method_name: str, baseline_mean: float) -> dict:
    values = df[f"precision_{method_name}"]
    mean_precision = values.mean()
    summary = {
        "method": method_name,
        "mean_precision": mean_precision,
        "median_precision": values.median(),
        "std_precision": values.std(),
        "n_periods": values.count(),
        "lift_vs_baseline": 0.0 if method_name == "baseline" else mean_precision / baseline_mean - 1,
        "diff_vs_baseline": np.nan,
        "diff_ci_low": np.nan,
        "diff_ci_high": np.nan,
        "random_precision": RANDOM_PRECISION,
    }
    if method_name != "baseline":
        diff = values - df["precision_baseline"]
        summary["diff_vs_baseline"] = diff.mean()
        summary["diff_ci_low"], summary["diff_ci_high"] = block_bootstrap_ci(diff)
    return summary


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    baseline_mean = df["precision_baseline"].mean()
    return pd.DataFrame([summarise_method(df, m, baseline_mean) for m in METHOD_NAMES])


def validate_summary(summary: pd.DataFrame) -> None:
    assert set(summary["method"]) == set(METHOD_NAMES)
    assert summary["n_periods"].gt(0).all()
    assert summary["mean_precision"].between(0, 1).all()
    assert summary["median_precision"].between(0, 1).all()
    assert summary["std_precision"].ge(0).all()
    print("Evaluation validation passed.")


if __name__ == "__main__":
    backtest_results = load_backtest_results(BACKTEST_RESULTS_PATH)
    summary_table = build_summary_table(backtest_results)
    validate_summary(summary_table)

    print("\nEvaluation summary:")
    for _, row in summary_table.iterrows():
        line = (f"  {row['method']}: mean={row['mean_precision']:.3f}, "
                f"median={row['median_precision']:.3f}, std={row['std_precision']:.3f}, "
                f"N={int(row['n_periods'])}, lift={row['lift_vs_baseline']:.1%}")
        if row["method"] != "baseline":
            line += (f", diff={row['diff_vs_baseline']:+.3f} "
                     f"[{row['diff_ci_low']:+.3f}, {row['diff_ci_high']:+.3f}]")
        print(line)
    print(f"  random selection: {RANDOM_PRECISION:.3f}")

    summary_table.to_csv(EVALUATION_RESULTS_PATH, index=False)
    print(f"\nWrote {EVALUATION_RESULTS_PATH}")
