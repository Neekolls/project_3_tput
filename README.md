# tput — Automated DataFrame Audit for ML

`tput` is a Python library for fast, opinionated data auditing before ML pipelines. Pass it a DataFrame and get a structured report that surfaces everything that matters before you start modelling: missing values, outliers, skewness, type issues, categorical noise, correlations, multicollinearity, and more.

```python
from tput import quick_report

report = quick_report(df, target="SalePrice")
report.show()       # full column-by-column report
report.summary()    # condensed overview for large datasets
```

---

## Installation

```bash
# Not yet on PyPI — clone and import locally
git clone https://github.com/yourname/tput
cd tput
pip install -e .
```

Dependencies: `pandas>=1.5`, `scipy>=1.10`, `scikit-learn>=1.0`

---

## Quickstart

```python
import pandas as pd
from tput import quick_report

df = pd.read_csv("titanic.csv")

# Basic audit
report = quick_report(df)
report.show()

# With target column — unlocks classification/regression analysis
report = quick_report(df, target="Survived")
report.show()

# Condensed view — useful for wide datasets (50+ columns)
report.summary()

# Programmatic access
report.get("nan_analysis")
report.get("target_analysis")
report.warnings
```

---

## What it detects

### Data quality
| Step | What it catches |
|---|---|
| `overview` | Shape, dtypes |
| `visible_missing` | NaN counts and percentages per column |
| `duplicates` | Fully duplicated rows |
| `categorical_profile` | Cardinality, mode, top/bottom values, hidden missing (`""`, `"   "`), case collisions (`"Paris"` vs `"paris"`) |
| `type_issues` | String columns that should be numeric or datetime |
| `row_analysis` | Rows with too many missing fields (default: ≥50%) — flagged for removal |

### Statistical analysis
| Step | What it catches |
|---|---|
| `numeric_profile` | mean, std, Q1/Q3, min/max, mode |
| `skewness` | Asymmetry level (symmetric / moderate / high) + recommended transform (log1p, sqrt, reflect + log1p, yeo-johnson) |
| `outliers` | IQR on symmetric columns, MAD on skewed ones — auto-selected per column |
| `nan_analysis` | MCAR / MAR / MNAR classification + imputation strategy (mean, median, mode, knn_or_regression). High missing rate (>70%) flagged as potential semantic NaN (e.g. "no pool", "no garage") |
| `feature_quality` | quasi-constant columns, low-cardinality numerics, suspected ID columns |

### Multicollinearity & associations
| Step | What it catches |
|---|---|
| `correlations` | Pearson pairs above threshold (default: 0.85) + Chi² / Cramér's V (bias-corrected) for categorical pairs |
| `vif` | Variance Inflation Factor per continuous column — flags multicollinearity for linear models |

### Target analysis (`target=` parameter)
When a target column is specified, `tput` automatically detects classification vs regression and runs:

**Classification:**
- Class balance and imbalance detection (minority class < 20% → warning)
- Feature correlation ranking (Cramér's V for categorical, point-biserial for numeric)
- Leakage detection (correlation > 0.95 with target)

**Regression:**
- Target skewness + recommended transform
- Outliers in the target (direct impact on loss function)
- Pearson correlation ranking for all numeric features
- Leakage detection

---

## Display modes

### `report.show()`
Column-by-column view — every column gets a block with all its properties stacked:

```
--- COLUMN: Age  [float64] ---
  nulls          : 177 (19.87%)
  mean / median  : 29.70 / 28.0  |  std: 14.53
  skewness       : 0.389 (symmetric)
  outliers       : 11 (1.54%)  [method=IQR  bounds=-6.69, 64.81]
  nan_analysis   : MAR -> impute (knn_or_regression)
    correlated_with: Pclass (r=0.173), Parch (r=-0.124)
  target_corr    : point_biserial_r=0.077 with 'Survived'
```

### `report.summary()`
Condensed grouped view — designed for datasets with 50+ columns where `show()` would be overwhelming:

```
=== TPUT SUMMARY ===
Shape : 1460 rows x 81 columns

ISSUES DETECTED:
  missing values   : 19 columns affected (6 proposed drop, 13 impute, 11 MAR)
  skewness         : 34 high, 12 moderate
  outliers         : 28 columns, 1842 values total
  feature quality  : 9 quasi-constant, 18 low_cardinality, 1 potential_id
  vif (redundant)  : 8 columns
  correlations     : 1 numeric pairs, 182 categorical pairs (showing top 10 in warnings)

Total warnings : 156  (use report.show() for full detail)
```

---

## Parameters

```python
quick_report(
    df,

    # Steps — toggle any on/off
    overview=True,
    visible_missing=True,
    duplicates=True,
    categorical_profile=True,
    type_issues=True,
    numeric_profile=True,
    skewness=True,
    outliers=True,
    nan_analysis=True,
    correlations=True,
    feature_quality=True,
    row_analysis=True,
    vif=True,

    # Target column — enables target_analysis
    target=None,                      # e.g. target="SalePrice" or target="Survived"

    # Thresholds
    nan_drop_threshold=0.30,          # drop column if missing rate > 30%
    correlation_threshold=0.85,       # Pearson |r| threshold for numeric pairs
    cramers_v_threshold=0.25,         # Cramér's V threshold for categorical pairs
    max_correlation_warnings=10,      # cap categorical warnings, show top N by V
    row_drop_threshold=0.50,          # flag rows with >= 50% NaN
    apply_row_filter=True,            # run outliers/nan/correlations on filtered df
    quasi_constant_threshold=0.95,    # flag column if dominant value > 95%
    low_cardinality_max_unique=10,    # numeric columns with <= N values → categorical
    vif_threshold=10.0,               # VIF threshold for multicollinearity warning

    # Display
    feature_display=True,             # True = column-by-column view, False = step-by-step
)
```

---

## Programmatic access

```python
# Access any step result directly
report.get("outliers")
report.get("nan_analysis")
report.get("skewness")
report.get("correlations")       # includes high_pairs and categorical_associations
report.get("target_analysis")    # classification or regression breakdown

# All warnings as a list
report.warnings

# Drop rows flagged by row_analysis
drop_idx = report.get("row_analysis")["rows_to_drop_idx"]
df_clean = df.drop(drop_idx)

# Full categorical association list (bypasses max_correlation_warnings cap)
all_cat_pairs = report.get("correlations")["categorical_associations"]

# Feature ranking by correlation with target
feature_corr = report.get("target_analysis")["feature_correlations"]
```

---

## Design decisions

**IQR vs MAD** — outlier method is chosen per column based on skewness. IQR is appropriate for symmetric distributions; MAD (Median Absolute Deviation) is more robust on skewed data where extreme values distort the mean and the IQR bounds.

**Bias-corrected Cramér's V** — categorical associations use the Bergsma (2013) bias correction, which removes the positive bias of the standard formula on small samples and low-cardinality columns. High-cardinality columns (>50% unique values) are excluded from Chi² computation as they produce artificially inflated V values.

**VIF via sklearn** — implemented without `statsmodels` using `LinearRegression` from sklearn. Only continuous columns (nunique > 10) are included. The target column is automatically excluded when `target=` is specified.

**Row filter** — `row_analysis` identifies sparse rows, then `outliers`, `nan_analysis`, `correlations`, and `vif` all run on the filtered DataFrame. This prevents sparse rows from distorting column-level statistics. The raw DataFrame is always used for the upstream steps (`overview`, `visible_missing`, `duplicates`, `categorical_profile`, `type_issues`, `numeric_profile`, `skewness`, `row_analysis` itself).

**Semantic NaN detection** — columns with >70% missing rate that are proposed for dropping receive an additional note: the high missing rate may encode absence of a feature (e.g. `PoolQC` missing = no pool) rather than truly missing data. Replacing NaN with a `"None"` category is often preferable to dropping.

**Target exclusion** — when `target=` is specified, the target column is automatically excluded from inter-feature correlation computation (both Pearson and Cramér's V). Feature-target relationships are computed separately in `target_analysis`.

---

## Known limitations

- **No domain knowledge.** A latitude of -78° or a weight of 1100 kg may be perfectly valid. All flagging is statistical.
- **Skewness on small samples.** Columns with <30 non-null values produce unreliable skewness estimates.
- **Outlier detection assumes unimodal distributions.** IQR and MAD both fail on bimodal distributions (e.g. a mixed-species weight column).
- **MNAR is a hypothesis.** The library flags it when missingness is high but unexplained — proving MNAR requires domain knowledge.
- **VIF assumes no perfect multicollinearity.** If the matrix is singular (e.g. `BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF = TotalBsmtSF` exactly), VIF is capped at 10000 to avoid division-by-zero. This is a signal, not a bug.
