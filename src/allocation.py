"""
Phase 7: Editorial allocation framework.

Ranks topics by the momentum score and allocates fixed editorial capacity
subject to a per-topic maximum. Decision-framework extension, not a claim
of predictive superiority.

Ties in score (common: many topics sit at exactly -1.00) are broken by
trailing 4-week activity, then by topic name, instead of by name alone.
"""

from pathlib import Path

import pandas as pd

from src.backtest import compute_baseline

INPUT_PATH = Path("data/processed/composite_features.csv")
OUTPUT_PATH = Path("data/processed/allocation_results.csv")

TOTAL_CAPACITY = 10.0
MAX_TOPIC_ALLOCATION = 3.0
EXPECTED_N_TOPICS = 16


def load_latest_valid_scores(path: Path) -> pd.DataFrame:
    """Load the latest week where all 16 topics have valid scores."""
    df = pd.read_csv(path, parse_dates=["week"])

    required = {"week", "topic", "activity", "momentum", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["topic"].nunique() != EXPECTED_N_TOPICS:
        raise ValueError(f"Expected {EXPECTED_N_TOPICS} topics, found {df['topic'].nunique()}.")

    # Tie-break column: trailing activity, computed on the full history.
    df = df.sort_values(["topic", "week"]).reset_index(drop=True)
    df["baseline"] = compute_baseline(df)

    valid = df.dropna(subset=["score", "baseline"]).copy()
    topic_counts = valid.groupby("week")["topic"].nunique()
    complete_weeks = topic_counts[topic_counts == EXPECTED_N_TOPICS].index
    if len(complete_weeks) == 0:
        raise ValueError("No week has valid scores for all 16 topics.")

    latest = valid[valid["week"] == complete_weeks.max()].copy()
    if len(latest) != EXPECTED_N_TOPICS:
        raise ValueError("Latest valid week does not contain exactly 16 topics.")
    return latest


def allocate_equal(topics: pd.DataFrame) -> pd.DataFrame:
    result = topics.copy()
    result["equal_allocation"] = TOTAL_CAPACITY / len(result)
    return result


def allocate_by_signal(topics: pd.DataFrame) -> pd.DataFrame:
    """Fill capacity in rank order (score, then trailing activity, then name)."""
    result = topics.sort_values(
        ["score", "baseline", "topic"], ascending=[False, False, True]
    ).reset_index(drop=True)

    remaining = TOTAL_CAPACITY
    allocations = []
    for _ in range(len(result)):
        allocation = min(MAX_TOPIC_ALLOCATION, remaining)
        allocations.append(allocation)
        remaining -= allocation
    result["signal_allocation"] = allocations
    return result


def calculate_hhi(allocations: pd.Series) -> float:
    total = allocations.sum()
    if total <= 0:
        raise ValueError("Allocation total must be positive.")
    return float(((allocations / total) ** 2).sum())


def validate_allocation(result: pd.DataFrame) -> None:
    assert len(result) == EXPECTED_N_TOPICS
    assert result["signal_allocation"].sum() == TOTAL_CAPACITY
    assert (result["signal_allocation"] >= 0).all()
    assert (result["signal_allocation"] <= MAX_TOPIC_ALLOCATION).all()
    assert abs(result["equal_allocation"].sum() - TOTAL_CAPACITY) < 1e-9
    print("Allocation validation passed.")


def main() -> None:
    topics = load_latest_valid_scores(INPUT_PATH)
    equal = allocate_equal(topics)
    signal = allocate_by_signal(topics)

    result = signal[["week", "topic", "activity", "momentum", "score", "signal_allocation"]].copy()
    # Merge by topic: `signal` is re-sorted, so row order differs from `equal`.
    result["equal_allocation"] = result["topic"].map(equal.set_index("topic")["equal_allocation"])

    validate_allocation(result)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWeek: {result['week'].iloc[0].date()}")
    print(result[["topic", "score", "signal_allocation"]].to_string(index=False))
    print(f"\nSignal HHI: {calculate_hhi(result['signal_allocation']):.3f}")
    print(f"Equal HHI:  {calculate_hhi(result['equal_allocation']):.3f}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
