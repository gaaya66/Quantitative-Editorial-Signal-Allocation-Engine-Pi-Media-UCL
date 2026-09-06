# Phase 1 -- Data Quality Report

```
Rows x Cols: 281 x 9
Date range: 2015-11-08 to 2026-07-25
Unparseable dates: 0
Fully-missing columns: ['topic', 'featured', 'url']
Duplicate article_id count: 0
Exact duplicate rows: 0
(title, author) duplicate pairs: 16

Missing value counts:
article_id      0
date            0
title           0
author          8
section         2
topic         281
tags           16
featured      281
url           281

Section value counts (raw, before normalisation):
section
Lifestyle & Culture    75
Opinion                63
News                   58
Features               26
Science and Tech       14
Editorial              13
Sport                  13
Sports                 12
Science & Tech          4
NaN                     2
Old Culture             1

Notes:
- 8 likely republished/duplicate articles found via (title, author) match -- see `duplicate_title_author_pairs`. These will be deduplicated in `clean_articles`, keeping the earliest publication date.
- Columns ['topic', 'featured', 'url'] are 100% missing in this export. 'topic' will be constructed in Phase 2 from tags/section. 'featured' and 'url' carry no information for the MVP and are dropped.
```

Duplicate (title, author) rows dropped during cleaning: 8

Final cleaned dataset: 273 rows, 2015-11-08 to 2026-07-25
