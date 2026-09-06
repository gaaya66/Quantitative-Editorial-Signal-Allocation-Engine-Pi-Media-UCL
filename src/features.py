"""
features.py
------------
Phase 2: builds the weekly topic activity panel used by all later
phases (rolling features, signal, backtest).

A(t, k) = number of articles about topic k in week t

Modelling period (fixed, per project decision): 2022-10-07 to
2026-06-29. This excludes two multi-year gaps found in the raw article
dates (2016-11-20 -> 2019-10-31 and 2021-09-05 -> 2022-10-07) that are
almost certainly archive/export gaps, not real editorial inactivity.

This module does NOT invent any topic classification of its own -- it
only calls `classify_tags()` from taxonomy.py, which is the single
source of truth for which topics an article's tags match.
"""

from __future__ import annotations

import pandas as pd

from src.taxonomy import classify_tags, ALL_TOPICS

MODELING_WINDOW_START = pd.Timestamp("2022-10-07")
MODELING_WINDOW_END = pd.Timestamp("2026-06-29")


def week_start(dates: pd.Series) -> pd.Series:
    """Map each date to the Monday that starts its ISO calendar week.

    Example: any date from Monday 2024-03-04 to Sunday 2024-03-10 is
    mapped to week_start = 2024-03-04.

    `dates.dt.dayofweek` gives Monday=0 ... Sunday=6, so subtracting
    that many days from each date always lands on the Monday of that
    date's week. This is the ONLY place the Monday convention is
    defined -- every other function in this project that needs a
    "week" goes through this function.
    """
    return dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")


def explode_articles_to_topics(cleaned_articles: pd.DataFrame) -> pd.DataFrame:
    """Turn one row per article into one row per (article, topic) pair.

    Uses `classify_tags()` from taxonomy.py -- this function does not
    do any classification itself, it only expands whatever
    `classify_tags()` returns. An article matching two topics
    contributes two rows here (one per topic). An article matching zero
    topics contributes NO rows and is excluded from the topic universe
    entirely -- it is never forced into a category.

    Expects `cleaned_articles` to have `article_id`, `date` (datetime),
    and `tags_list` (already parsed into a Python list of strings).

    Returns a DataFrame with columns: article_id, week, topic.
    """
    records = []
    for row in cleaned_articles.itertuples():
        matched_topics = classify_tags(row.tags_list)
        for topic in matched_topics:
            records.append({"article_id": row.article_id, "date": row.date, "topic": topic})

    exploded = pd.DataFrame(records, columns=["article_id", "date", "topic"])
    exploded["week"] = week_start(exploded["date"])

    return exploded[["article_id", "week", "topic"]]


def build_weekly_topic_panel(
    article_topic_pairs: pd.DataFrame,
    window_start: pd.Timestamp = MODELING_WINDOW_START,
    window_end: pd.Timestamp = MODELING_WINDOW_END,
) -> pd.DataFrame:
    """Build the complete weekly x topic activity panel over the fixed
    modelling period, with explicit zero rows for topic-weeks that had
    no classified articles.

    Why explicit zeros matter: if we only kept rows produced by
    `groupby(["week", "topic"]).size()`, a topic-week with genuinely
    ZERO articles would not appear in the output at all -- it would be
    MISSING, not zero. A rolling mean or standard deviation computed
    later would then treat that week as absent rather than correctly
    pulling the average down to reflect real editorial silence. Only an
    explicit zero row (via `reindex(..., fill_value=0)` below) avoids
    this.

    The week x topic grid is built from the FIXED modelling period
    boundaries (not from whatever dates happen to appear in the data),
    so the panel always has exactly the same shape regardless of which
    weeks happened to have articles.

    Returns a long-format DataFrame with columns: week, topic, activity.
    """
    assert len(ALL_TOPICS) == 16, f"Expected 16 frozen topics, found {len(ALL_TOPICS)}."

    first_week = week_start(pd.Series([window_start])).iloc[0]
    last_week = week_start(pd.Series([window_end])).iloc[0]

    in_window = (article_topic_pairs["week"] >= first_week) & (article_topic_pairs["week"] <= last_week)
    pairs_in_window = article_topic_pairs[in_window]

    weekly_counts = pairs_in_window.groupby(["week", "topic"]).size()
    weekly_counts = weekly_counts.rename("activity")

    all_weeks = pd.date_range(start=first_week, end=last_week, freq="W-MON")
    full_index = pd.MultiIndex.from_product([all_weeks, ALL_TOPICS], names=["week", "topic"])

    panel = weekly_counts.reindex(full_index, fill_value=0)
    panel = panel.reset_index()

    _validate_weekly_panel(panel, all_weeks)

    return panel


def _validate_weekly_panel(panel: pd.DataFrame, expected_weeks: pd.DatetimeIndex) -> None:
    """Sanity checks on the finished panel. Raises an AssertionError
    immediately if any of these fail, rather than letting a broken
    panel silently flow into later phases.
    """
    assert set(panel["topic"].unique()) == set(ALL_TOPICS), (
        "Panel does not contain exactly the 16 frozen topics."
    )

    assert not panel.duplicated(subset=["week", "topic"]).any(), (
        "Duplicate (week, topic) rows found -- grid should be unique per pair."
    )

    weeks_per_topic = panel.groupby("topic").size()
    assert (weeks_per_topic == len(expected_weeks)).all(), (
        "Every topic must have exactly one row for every week in the modelling period."
    )

    topics_per_week = panel.groupby("week").size()
    assert (topics_per_week == len(ALL_TOPICS)).all(), (
        "Every week must have exactly one row for every one of the 16 topics."
    )

    assert (panel["activity"] >= 0).all(), "Activity counts must never be negative."
    assert panel["activity"].dtype.kind in "iu", "Activity counts must be integers."