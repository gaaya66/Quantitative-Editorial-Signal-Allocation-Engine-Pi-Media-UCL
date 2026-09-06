"""
Phase 7: Editorial allocation framework.

Uses the final momentum signal to rank topics and allocate a fixed
editorial capacity subject to a maximum allocation per topic.

This is a decision-framework extension, not a claim of predictive
superiority.
"""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/composite_features.csv")
OUTPUT_PATH = Path("data/processed/allocation_results.csv")

TOTAL_CAPACITY = 10.0
MAX_TOPIC_ALLOCATION = 3.0
EXPECTED_N_TOPICS = 16


def load_latest_valid_scores(path: Path) -> pd.DataFrame:
    """Load the latest week where all 16 topics have valid scores."""

    df = pd.read_csv(path, parse_dates=["week"])

    required_columns = {
        "week",
        "topic",
        "activity",
        "momentum",
        "score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["topic"].nunique() != EXPECTED_N_TOPICS:
        raise ValueError(
            f"Expected {EXPECTED_N_TOPICS} topics, "
            f"found {df['topic'].nunique()}."
        )

    valid = df.dropna(subset=["score"]).copy()

    topic_counts = valid.groupby("week")["topic"].nunique()

    complete_weeks = topic_counts[
        topic_counts == EXPECTED_N_TOPICS
    ].index

    if len(complete_weeks) == 0:
        raise ValueError(
            "No week has valid scores for all 16 topics."
        )

    latest_week = complete_weeks.max()

    latest = valid[
        valid["week"] == latest_week
    ].copy()

    if len(latest) != EXPECTED_N_TOPICS:
        raise ValueError(
            "Latest valid week does not contain exactly 16 topics."
        )

    return latest


def allocate_equal(topics: pd.DataFrame) -> pd.DataFrame:
    """Allocate capacity equally across all topics."""

    result = topics.copy()

    result["equal_allocation"] = (
        TOTAL_CAPACITY / len(result)
    )

    return result


def allocate_by_signal(topics: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate capacity according to momentum signal rank.

    Topics are ranked by score and assigned capacity sequentially,
    subject to the maximum allocation per topic.
    """

    result = topics.copy()

    result = result.sort_values(
        ["score", "topic"],
        ascending=[False, True],
    ).reset_index(drop=True)

    remaining_capacity = TOTAL_CAPACITY
    allocations = []

    for _ in range(len(result)):
        allocation = min(
            MAX_TOPIC_ALLOCATION,
            remaining_capacity,
        )

        allocations.append(allocation)
        remaining_capacity -= allocation

    result["signal_allocation"] = allocations

    return result


def calculate_hhi(allocations: pd.Series) -> float:
    """Calculate the Herfindahl-Hirschman Index."""

    total = allocations.sum()

    if total <= 0:
        raise ValueError(
            "Allocation total must be positive."
        )

    shares = allocations / total

    return float((shares ** 2).sum())


def validate_allocation(result: pd.DataFrame) -> None:
    """Validate the final allocation output."""

    assert len(result) == EXPECTED_N_TOPICS, (
        f"Expected {EXPECTED_N_TOPICS} topics."
    )

    assert result["signal_allocation"].sum() == TOTAL_CAPACITY, (
        "Signal allocation does not use the full capacity."
    )

    assert (
        result["signal_allocation"] >= 0
    ).all(), "Negative signal allocation found."

    assert (
        result["signal_allocation"] <= MAX_TOPIC_ALLOCATION
    ).all(), "Maximum topic allocation exceeded."

    assert (
        result["equal_allocation"].sum()
        == TOTAL_CAPACITY
    ), "Equal allocation does not use the full capacity."

    print("Allocation validation passed:")
    print(f"  - {len(result)} topics")
    print("  - full capacity allocated")
    print("  - no topic exceeds maximum allocation")
    print("  - equal-allocation benchmark validated")


def main() -> None:
    topics = load_latest_valid_scores(INPUT_PATH)

    equal = allocate_equal(topics)

    signal = allocate_by_signal(topics)

    result = signal[
        [
            "week",
            "topic",
            "activity",
            "momentum",
            "score",
            "signal_allocation",
        ]
    ].copy()

    result["equal_allocation"] = (
        equal["equal_allocation"].values
    )

    validate_allocation(result)

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    signal_hhi = calculate_hhi(
        result["signal_allocation"]
    )

    equal_hhi = calculate_hhi(
        result["equal_allocation"]
    )

    print()
    print("Allocation analysis:")
    print(
        f"Week: {result['week'].iloc[0].date()}"
    )
    print(
        f"Topics: {len(result)}"
    )
    print(
        f"Total capacity: "
        f"{result['signal_allocation'].sum():.1f}"
    )
    print(
        f"Maximum allocation per topic: "
        f"{MAX_TOPIC_ALLOCATION:.1f}"
    )

    print()
    print("Signal-ranked allocation:")
    print(
        result[
            ["topic", "score", "signal_allocation"]
        ].to_string(index=False)
    )

    print()
    print(
        f"Signal allocation HHI: {signal_hhi:.3f}"
    )
    print(
        f"Equal allocation HHI:  {equal_hhi:.3f}"
    )

    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()