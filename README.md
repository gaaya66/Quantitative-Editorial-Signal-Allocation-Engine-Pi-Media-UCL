# Quantitative-Editorial-Signal-Allocation-Engine-Pi-Media-UCL
# Quantitative Editorial Signal & Allocation Engine

**Pi Media, UCL**

A quantitative research project testing whether recent editorial activity can be used to predict which topics will receive higher coverage in the following week.

The project builds weekly topic-level time series from Pi Media's article archive, engineers a momentum signal, evaluates it using walk-forward backtesting, tests sensitivity to different parameters, and applies the resulting rankings to a simple editorial allocation framework.

## Research Question

**Does recent activity in a topic contain information about its subsequent editorial coverage?**

The main signal measures recent activity relative to a longer trailing average. It is compared against a simple baseline based only on recent article counts.

---

## Data

The raw dataset contains **281 articles** from Pi Media, covering 2015–2026.

After removing duplicate `(title, author)` observations, **273 articles** remain.

The archive is unevenly distributed over time, with **163 of the 281 raw articles from 2026**. Because earlier periods contain relatively little data, the modelling period focuses on the more usable section of the archive.

The resulting panel contains:

* 196 weeks
* 16 topics
* 3,136 topic-week observations
* 245 classified articles
* 28 unclassified articles

Weeks with no articles for a topic are kept as zero observations rather than being removed.

---

## Topic Classification

Articles are mapped to a fixed taxonomy of 16 topics:

* UCL & Campus
* Society & Identity
* UK Politics
* Theatre & Visual Arts
* US & International Politics
* Middle East & Global Conflict
* Science & Technology
* Sport & Varsity
* Film & Television
* Climate & Environment
* Health & Medicine
* Lifestyle, Food & Travel
* Global Affairs & Geopolitics
* Art & Culture
* Economics & Finance
* Education

The topic universe was fixed before running the backtest.

Music was excluded because only five articles were classified into the category and activity was concentrated in a short period. This was decided before evaluating model performance.

---

## Signal

For topic \(k\) in week \(t\), weekly activity is:

$$
A_{t,k} = \text{number of articles about topic } k
$$

The main signal is a momentum measure:

$$
Momentum_{t,k}
=
\frac{
\text{mean activity over the previous 4 weeks}
}{
\text{mean activity over the previous 12 weeks}+\epsilon
}
-1
$$

A positive value means recent activity is above the longer-term average.

The primary specification uses a **4-week / 12-week** window.

All features are calculated using information available at the prediction date.

### Z-score experiment

I also tested a rolling z-score to measure abnormal activity relative to historical volatility.

This was not used in the final model because the data are too sparse. A large proportion of topic-week windows have zero or near-zero variance, making the resulting z-scores unstable.

Rather than adding arbitrary regularisation to force the measure to work, the signal was dropped.

---

## Backtest

The model is evaluated using a **walk-forward** procedure.

For each week:

1. Calculate the signal using information available up to that week.
2. Rank the 16 topics.
3. Take the five highest-ranked topics.
4. Look at the actual five highest-activity topics in the following week.
5. Calculate Precision@5.

$$
Precision@5 =
\frac{
|\text{Predicted Top 5} \cap \text{Actual Top 5}|
}{5}
$$

This avoids using future observations when constructing the signal.

### Baseline

The benchmark ranks topics using their **trailing 4-week average activity**.

This is deliberately simple: the momentum signal needs to add information beyond the fact that recently active topics are likely to remain active.

---

## Results

The main backtest covers **184 weekly evaluation periods**.

| Method                     | Mean Precision@5 |
| -------------------------- | ---------------: |
| Trailing activity baseline |        **70.5%** |
| 4/12 momentum              |        **68.8%** |

The momentum signal therefore **did not outperform the baseline**.

The difference is approximately **1.7 percentage points**, with the baseline performing better.

This result was kept rather than tuning the signal after seeing the backtest.

---

## Robustness

I tested three momentum specifications:

| Short | Long | Evaluation periods | Precision@5 | Lift vs baseline |
| ----: | ---: | -----------------: | ----------: | ---------------: |
|     4 |   12 |                184 |       68.8% |            −1.6% |
|     6 |   18 |                178 |       65.1% |            −6.9% |
|     8 |   24 |                172 |       63.1% |            −9.7% |

The baseline achieved **69.9% Precision@5** over the 192 periods used in the robustness analysis.

Momentum underperformed the baseline for all three window choices.

---

## Allocation Framework

As an extension, the signal rankings are converted into a simple constrained allocation.

Parameters:

* Total capacity: **10 units**
* Maximum per topic: **3 units**
* 16 topics

The highest-ranked topics receive capacity first until the total capacity is used.

At the latest modelling week, the resulting allocation was:

| Topic                 | Momentum | Allocation |
| --------------------- | -------: | ---------: |
| Science & Technology  |     2.00 |          3 |
| Theatre & Visual Arts |    −0.25 |          3 |
| Art & Culture         |    −1.00 |          3 |
| Climate & Environment |    −1.00 |          1 |
| All other topics      |    −1.00 |          0 |

This is intended as an example of how a quantitative ranking could be used in an editorial decision framework, rather than as an optimisation result.

---

## Limitations

* The Pi Media archive is incomplete and heavily concentrated in 2026.
* Some topics have long periods with zero activity.
* The topic taxonomy involves manual classification rules.
* The dataset contains only 16 topics, limiting the size of the cross-section.
* Editorial priorities can change over time, so historical relationships may not remain stable.
* The allocation framework is illustrative and has not been tested as a real editorial decision-making system.

---

## Why a Simple Signal?

The dataset is relatively small and sparse, so more complex machine learning models would have a high risk of overfitting.

The aim of this project was therefore to first test whether a simple, interpretable time-series signal contained useful information before introducing more complex models.

The current results suggest that **recent activity itself is a better predictor than the momentum measure tested here**.

---

## Project Structure

```text
data/
├── raw/
└── processed/

src/
├── data.py
├── taxonomy.py
├── features.py
├── signals.py
├── composite.py
├── backtest.py
├── evaluation.py
├── robustness.py
├── allocation.py
└── visualisation.py

reports/
```

## Tools

* Python
* pandas
* NumPy
* Matplotlib
* rolling time-series features
* walk-forward backtesting
* quantitative signal evaluation
* constrained allocation
