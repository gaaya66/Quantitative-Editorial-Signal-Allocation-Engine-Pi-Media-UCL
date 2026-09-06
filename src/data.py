"""
data.py
-------
Phase 1 of the Quantitative Editorial Signal Research Engine.

Responsibilities of this module:
    1. Load the raw Pi Media article export exactly as scraped/exported.
    2. Validate structural assumptions about the raw data (no silent
       corruption of IDs, dates, etc.) before any cleaning happens.
    3. Clean the data: standardise inconsistent categorical labels,
       resolve duplicate publications, parse tags into a usable list
       structure, and make missingness explicit rather than implicit.

Design principle: raw data is NEVER mutated in place. `load_raw_articles`
returns exactly what is on disk (modulo dtype parsing). `clean_articles`
takes that frame and returns a NEW, cleaned frame. This mirrors how you
would treat a point-in-time data vendor feed in a real quant pipeline --
you keep the untouched vendor snapshot, and cleaning is a separate,
auditable, reproducible transformation on top of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "article_id", "date", "title", "author", "section",
    "topic", "tags", "featured", "url",
]

# Sections that are semantically identical but spelled inconsistently in
# the raw export. Mapping is derived from actually inspecting
# df['section'].value_counts() -- not guessed a priori.
SECTION_NORMALISATION = {
    "Sports": "Sport",
    "Science & Tech": "Science and Tech",
    # 'Old Culture' is a single legacy row (2020) from what is now the
    # Lifestyle & Culture section under an old CMS label.
    "Old Culture": "Lifestyle & Culture",
}


@dataclass
class DataQualityReport:
    """Container for everything Phase 1 needs to know about the raw data.

    Populated by `validate_raw`. Kept as a dataclass (not printed
    immediately) so the same report can be logged, asserted on in tests,
    or written to a markdown file.
    """
    n_rows: int
    n_cols: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    missing_counts: pd.Series
    fully_missing_columns: list[str]
    n_duplicate_article_ids: int
    n_exact_duplicate_rows: int
    duplicate_title_author_pairs: pd.DataFrame
    section_value_counts: pd.Series
    unparseable_dates: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Rows x Cols: {self.n_rows} x {self.n_cols}",
            f"Date range: {self.date_min.date()} to {self.date_max.date()}",
            f"Unparseable dates: {self.unparseable_dates}",
            f"Fully-missing columns: {self.fully_missing_columns}",
            f"Duplicate article_id count: {self.n_duplicate_article_ids}",
            f"Exact duplicate rows: {self.n_exact_duplicate_rows}",
            f"(title, author) duplicate pairs: {len(self.duplicate_title_author_pairs)}",
            "",
            "Missing value counts:",
            self.missing_counts.to_string(),
            "",
            "Section value counts (raw, before normalisation):",
            self.section_value_counts.to_string(),
        ]
        if self.notes:
            lines += ["", "Notes:"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def load_raw_articles(path: str) -> pd.DataFrame:
    """Load the raw Pi Media export with explicit, honest dtypes.

    We deliberately do NOT parse dates here with `parse_dates=` in
    `read_csv`, because we want `validate_raw` to be able to measure how
    many dates fail to parse under an explicit, controlled conversion
    (silent coercion at load time would hide that diagnostic).

    Parameters
    ----------
    path : str
        Path to the raw CSV (e.g. data/raw/pi_media_articles_real.csv).

    Returns
    -------
    pd.DataFrame
        Raw article data, column order preserved, no rows dropped.
    """
    df = pd.read_csv(path, dtype={"article_id": str})
    missing_cols = set(RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw file is missing expected columns: {missing_cols}")
    return df


def validate_raw(df: pd.DataFrame) -> DataQualityReport:
    """Run structural diagnostics on the raw article data.

    This function makes NO changes to `df` -- it only measures. The
    resulting `DataQualityReport` is what Phase 1's cleaning decisions
    (Section normalisation, dedup policy, missing-value handling) are
    justified against, so every cleaning choice downstream is traceable
    to a specific, printed diagnostic rather than an assumption.

    What we check and why it matters for a point-in-time research
    pipeline:
      - Duplicate article_id: would silently corrupt joins between raw
        articles and any per-article feature computed downstream.
      - Exact duplicate rows / (title, author) duplicate pairs: article
        exports scraped incrementally are prone to re-ingesting the same
        publication (e.g. after a section correction). Left unhandled,
        these double-count activity in a topic-week, biasing Z-scores
        and momentum upward for no real editorial reason.
      - Unparseable dates: any week-level time series is only as
        trustworthy as its date parsing. A single bad date can silently
        drop or misplace an article's week bucket.
      - Fully-missing columns (topic, featured, url in this export):
        confirms these columns carry zero information in the current
        data and must be constructed (topic) or dropped (featured, url)
        rather than "cleaned".
    """
    notes: list[str] = []

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    n_unparseable = int(parsed_dates.isna().sum())

    missing_counts = df.isna().sum()
    fully_missing_columns = missing_counts[missing_counts == len(df)].index.tolist()

    n_dup_ids = int(df["article_id"].duplicated().sum())
    n_exact_dupes = int(df.duplicated().sum())

    match_key = df["title"].apply(_normalise_title_for_matching) + "||" + df["author"].fillna("")
    dup_mask = match_key.duplicated(keep=False) & df["author"].notna()
    dup_pairs = df.loc[dup_mask, ["article_id", "date", "title", "author", "section"]].sort_values(
        ["title", "date"]
    )

    if n_dup_ids > 0:
        notes.append(
            f"{n_dup_ids} duplicate article_id(s) found -- this breaks the "
            "one-row-per-article assumption and must be resolved before Phase 3."
        )
    if len(dup_pairs) > 0:
        notes.append(
            f"{len(dup_pairs) // 2 if len(dup_pairs) % 2 == 0 else len(dup_pairs)} "
            "likely republished/duplicate articles found via (title, author) match "
            "-- see `duplicate_title_author_pairs`. These will be deduplicated in "
            "`clean_articles`, keeping the earliest publication date."
        )
    if fully_missing_columns:
        notes.append(
            f"Columns {fully_missing_columns} are 100% missing in this export. "
            "'topic' will be constructed in Phase 2 from tags/section. "
            "'featured' and 'url' carry no information for the MVP and are dropped."
        )

    return DataQualityReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        date_min=parsed_dates.min(),
        date_max=parsed_dates.max(),
        missing_counts=missing_counts,
        fully_missing_columns=fully_missing_columns,
        n_duplicate_article_ids=n_dup_ids,
        n_exact_duplicate_rows=n_exact_dupes,
        duplicate_title_author_pairs=dup_pairs,
        section_value_counts=df["section"].value_counts(dropna=False),
        unparseable_dates=n_unparseable,
        notes=notes,
    )


def _normalise_title_for_matching(title: str) -> str:
    """Normalise a title for DUPLICATE-DETECTION purposes only.

    This does not touch the `title` column that gets stored -- it is
    used to build a throwaway matching key. It collapses Unicode
    punctuation variants (curly vs straight quotes, en/em dashes) and
    whitespace so that two publications of the same article are
    recognised as duplicates even when one was typed with a straight
    apostrophe and the other with a typographic one -- exactly what was
    found in the real data (PI0210 vs PI0211, "America's" vs
    "America's" footnote apostrophe).
    """
    t = title
    for a, b in [("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2013", "-"), ("\u2014", "-")]:
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip().lower()


def _split_tags(raw_tags: str | float) -> list[str]:
    """Parse a raw tags string into a clean list of individual tags.

    The raw export mixes comma- and semicolon-separated tag lists (both
    appear in the real data, e.g. "Sport, Varsity" vs
    "2024 US Presidential Elections; US Politics; Trump"), and has
    inconsistent capitalisation and stray whitespace. This function
    normalises the SEPARATOR and whitespace but deliberately does NOT
    force a single casing convention or merge synonyms (e.g. "politics"
    vs "Politics", "climate crisis" vs "Climate Crisis") -- that
    controlled-vocabulary mapping is Phase 2's job (topic taxonomy),
    not Phase 1's (structural cleaning). Collapsing synonyms here would
    hide exactly the messiness Phase 2 is supposed to demonstrate
    handling.

    Returns an empty list for missing/NaN tag fields.
    """
    if pd.isna(raw_tags):
        return []
    parts = re.split(r"[;,]", str(raw_tags))
    return [p.strip() for p in parts if p.strip()]


def clean_articles(df: pd.DataFrame, report: DataQualityReport | None = None) -> pd.DataFrame:
    """Produce a cleaned, analysis-ready article table from raw data.

    Cleaning steps (each justified by `validate_raw` diagnostics):
      1. Parse `date` to datetime; assert zero unparseable dates (the
         real export has none, but this is asserted rather than assumed
         so any future data pull that breaks this fails loudly).
      2. Normalise `section` labels (Sports -> Sport, etc.) using a
         mapping derived from inspecting the actual raw value_counts.
      3. Fill missing `section` (2 rows) with an explicit "Unknown"
         category rather than silently dropping the rows -- dropping
         would lose real editorial activity from the weekly counts.
      4. Fill missing `author` (8 rows) with an explicit "Unknown"
         label for the same reason.
      5. Parse `tags` into a list column (`tags_list`), preserving the
         ORIGINAL raw tag strings for provenance -- nothing here is
         thrown away.
      6. Drop `featured` and `url`, which are 100% missing in this
         export and carry no information.
      7. Deduplicate near-identical (title, author) publications,
         keeping the earliest `date`. This matters because an
         unresolved duplicate double-counts a single piece of editorial
         output as two -- inflating that topic's weekly activity for a
         reason that has nothing to do with editorial signal.
      8. Sort by date and reset the index, which the downstream weekly
         aggregation and rolling-window logic assumes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw article data, as returned by `load_raw_articles`.
    report : DataQualityReport, optional
        If not provided, `validate_raw(df)` is called internally.

    Returns
    -------
    pd.DataFrame
        Cleaned data with columns:
        [article_id, date, title, author, section, tags_raw, tags_list]
    """
    if report is None:
        report = validate_raw(df)

    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="raise")
    assert out["date"].isna().sum() == 0, "Unexpected unparseable dates after cleaning."

    out["section"] = out["section"].replace(SECTION_NORMALISATION)
    out["section"] = out["section"].fillna("Unknown")

    out["author"] = out["author"].fillna("Unknown")

    out["tags_raw"] = out["tags"]
    out["tags_list"] = out["tags"].apply(_split_tags)

    out = out.drop(columns=["topic", "featured", "url", "tags"])

    # Deduplicate: same (title, author) republished under a corrected
    # section / on a nearby date -- including cases that differ only by
    # Unicode punctuation (e.g. a straight vs. curly apostrophe), which
    # a naive exact-string match on (title, author) misses entirely.
    # Keep the earliest occurrence, which reflects the true
    # first-publication date of the editorial content.
    before = len(out)
    match_key = out["title"].apply(_normalise_title_for_matching) + "||" + out["author"]
    out = (
        out.assign(_match_key=match_key)
        .sort_values("date")
        .drop_duplicates(subset="_match_key", keep="first")
        .drop(columns="_match_key")
    )
    n_dropped = before - len(out)

    out = out.sort_values("date").reset_index(drop=True)

    # Structural assertions -- these are the guardrails that catch a
    # broken future data pull rather than letting it silently flow into
    # the signal-construction phase.
    assert out["article_id"].is_unique, "article_id is not unique after cleaning."
    assert out["date"].is_monotonic_increasing, "Dates are not sorted after cleaning."
    remaining_key = out["title"].apply(_normalise_title_for_matching) + "||" + out["author"]
    assert not remaining_key.duplicated().any(), (
        "Duplicate (title, author) pairs remain after deduplication."
    )

    out.attrs["n_duplicates_dropped"] = n_dropped
    return out