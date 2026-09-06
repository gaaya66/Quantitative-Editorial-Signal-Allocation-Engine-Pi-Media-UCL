"""
run_phase1.py
-------------
Entry point for Phase 1: inspect the raw Pi Media article export, produce
a data-quality report, clean the data, and write the cleaned table to
data/processed/. Run from the project root:

    python scripts/run_phase1.py
"""

import sys
from pathlib import Path

# allow running as `python scripts/run_phase1.py` from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import load_raw_articles, validate_raw, clean_articles

RAW_PATH = "data/raw/pi_media_articles.csv"
PROCESSED_PATH = "data/processed/cleaned_articles.csv"
REPORT_PATH = "reports/01_data_quality_report.md"


def main() -> None:
    df_raw = load_raw_articles(RAW_PATH)
    report = validate_raw(df_raw)

    print(report.summary())

    df_clean = clean_articles(df_raw, report=report)
    n_dropped = df_clean.attrs.get("n_duplicates_dropped", 0)
    print(f"\nDropped {n_dropped} duplicate (title, author) rows during cleaning.")
    print(f"Clean dataset: {len(df_clean)} rows, {df_clean['date'].min().date()} "
          f"to {df_clean['date'].max().date()}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(PROCESSED_PATH, index=False)

    with open(REPORT_PATH, "w") as f:
        f.write("# Phase 1 -- Data Quality Report\n\n")
        f.write("```\n")
        f.write(report.summary())
        f.write("\n```\n\n")
        f.write(f"Duplicate (title, author) rows dropped during cleaning: {n_dropped}\n\n")
        f.write(f"Final cleaned dataset: {len(df_clean)} rows, "
                f"{df_clean['date'].min().date()} to {df_clean['date'].max().date()}\n")

    print(f"\nWrote cleaned data to {PROCESSED_PATH}")
    print(f"Wrote data quality report to {REPORT_PATH}")


if __name__ == "__main__":
    main()