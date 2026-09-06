"""
run_phase2.py
-------------
Phase 2: build the weekly topic activity panel from the cleaned
articles, using the frozen 16-topic taxonomy.

    data/processed/cleaned_articles.csv
            |  classify_tags()              (src/taxonomy.py)
            |  explode_articles_to_topics()  (src/features.py)
            |  build_weekly_topic_panel()    (src/features.py)
            v
    data/processed/weekly_topic_panel.csv

Run from the project root:
    python scripts/run_phase2.py
"""

import ast
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.taxonomy import ALL_TOPICS
from src.features import (
    explode_articles_to_topics,
    build_weekly_topic_panel,
    MODELING_WINDOW_START,
    MODELING_WINDOW_END,
)

CLEANED_PATH = "data/processed/cleaned_articles.csv"
WEEKLY_PANEL_PATH = "data/processed/weekly_topic_panel.csv"


def main() -> None:
    cleaned_articles = pd.read_csv(CLEANED_PATH, parse_dates=["date"])
    cleaned_articles["tags_list"] = cleaned_articles["tags_list"].apply(ast.literal_eval)

    n_cleaned_articles = len(cleaned_articles)

    article_topic_pairs = explode_articles_to_topics(cleaned_articles)
    n_classified_articles = article_topic_pairs["article_id"].nunique()
    n_unclassified_articles = n_cleaned_articles - n_classified_articles

    weekly_panel = build_weekly_topic_panel(
        article_topic_pairs,
        window_start=MODELING_WINDOW_START,
        window_end=MODELING_WINDOW_END,
    )

    n_topics = weekly_panel["topic"].nunique()
    n_weeks = weekly_panel["week"].nunique()
    n_panel_rows = len(weekly_panel)
    first_week = weekly_panel["week"].min()
    last_week = weekly_panel["week"].max()
    total_activity = int(weekly_panel["activity"].sum())

    every_combination_present = n_panel_rows == n_weeks * n_topics

    print(f"Cleaned article count:       {n_cleaned_articles}")
    print(f"Classified article count:    {n_classified_articles}")
    print(f"Unclassified article count:  {n_unclassified_articles}")
    print()
    print(f"Number of topics:            {n_topics}")
    print(f"Number of weeks:             {n_weeks}")
    print(f"Total panel rows:            {n_panel_rows}")
    print(f"First week:                  {first_week.date()}")
    print(f"Last week:                   {last_week.date()}")
    print(f"Every week x topic present:  {every_combination_present} "
          f"({n_weeks} weeks x {n_topics} topics = {n_weeks * n_topics} expected rows)")
    print(f"Total activity (article-topic observations): {total_activity}")

    assert "Music" not in weekly_panel["topic"].unique(), "Music must not appear in the frozen taxonomy."
    assert n_topics == 16, f"Expected exactly 16 frozen topics, found {n_topics}."

    weekly_panel.to_csv(WEEKLY_PANEL_PATH, index=False)
    print(f"\nWrote {WEEKLY_PANEL_PATH}")


if __name__ == "__main__":
    main()