# AI Classifier Controls Expansion Design

**Date:** 2026-03-17
**Scope:** Add hyperparameters and data preprocessing options to module4_rf.py

## Problem

The AI Flood Classifier only has 4 adjustable elements (feature selection, n_trees, max_depth, test events). Students need more knobs to experiment with and understand how ML model tuning and data preprocessing affect performance.

## Design

### 1. Additional Hyperparameters

Added to the existing "Model Parameters" container, below the current controls:

| Parameter | UI | Range | Default | Student-facing description |
|-----------|-----|-------|---------|---------------------------|
| `min_samples_leaf` | Slider | 1-20 | 1 | "Min samples per leaf node — larger = less overfitting" |
| `max_features` | Selectbox | sqrt / log2 / all | sqrt | "Features considered per split — diversity vs accuracy" |
| `class_weight` | Checkbox | balanced / None | None | "Auto-weight correction when flood samples are rare" |
| `bootstrap` | Checkbox | on / off | on | "Each tree trains on a random subset of data" |

These map directly to `sklearn.ensemble.RandomForestClassifier` constructor arguments.

### 2. Data Preprocessing Section

A new "Data Preprocessing" container placed ABOVE the "Model Parameters" container:

| Option | UI | Choices | Default | Description |
|--------|-----|---------|---------|-------------|
| Feature Scaling | Selectbox | None / StandardScaler / MinMaxScaler | None | "Normalize value ranges — RF usually doesn't need this, but experiment!" |
| Class Balance | Selectbox | None / Oversample minority / Undersample majority | None | "Handle flood/non-flood sample imbalance" |
| Outlier Removal | Selectbox | None / IQR method / Z-score (>3sigma) | None | "Remove extreme values — how does it affect accuracy?" |
| Sample Size | Slider | 10%-100%, step 10 | 100% | "Use partial data — see how data volume affects performance" |

### 3. Implementation Details

#### Preprocessing pipeline (applied before train/test split):

1. **Sample Size**: Random subsample of combined DataFrame (stratified by label)
2. **Outlier Removal**:
   - IQR: Remove rows where any feature value is outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
   - Z-score: Remove rows where any feature has |z| > 3
3. **Class Balance** (applied to training set only, after event-based split):
   - Oversample minority: Duplicate flood samples to match non-flood count
   - Undersample majority: Randomly reduce non-flood samples to match flood count
4. **Feature Scaling** (fit on training set, transform both train and test):
   - StandardScaler: zero mean, unit variance
   - MinMaxScaler: scale to [0, 1]

#### New hyperparameters passed to RandomForestClassifier:

```python
clf = RandomForestClassifier(
    n_estimators=n_trees,
    max_depth=max_depth if max_depth > 0 else None,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,  # "sqrt", "log2", or None (all)
    class_weight="balanced" if use_class_weight else None,
    bootstrap=bootstrap,
    random_state=seed, n_jobs=-1,
)
```

### 4. UI Layout

```
┌─ Data Preprocessing ──────────┐
│ Feature Scaling: [None ▼]     │
│ Class Balance:   [None ▼]     │
│ Outlier Removal: [None ▼]     │
│ Sample Size:     [===100%===] │
└───────────────────────────────┘
┌─ Model Parameters ───────────┐
│ Features: ☑SAR ☑NDWI ...     │
│ Trees:        [===100===]    │
│ Max Depth:    [====5====]    │
│ Min Leaf:     [====1====]    │
│ Max Features: [sqrt ▼]       │
│ ☑ Bootstrap  ☐ Class Weight  │
│ Test Events:  [multiselect]  │
└───────────────────────────────┘
        [🚀 Train AI Model!]
```

### 5. Metric display update

After training, show a small info line about preprocessing applied:

```
Trained on: Harvey, Pakistan (1,200 samples) · Tested on: Dubai (400 samples) · 0.8s
Preprocessing: StandardScaler · Oversample minority · 80% sample
```

## Files Changed

| File | Changes |
|------|---------|
| `modules/module4_rf.py` | New preprocessing UI + logic, additional hyperparameter controls, updated `train_rf` function |

## Out of Scope

- Other modules (SAR, Optical, Rainfall)
- Adding new ML algorithms (e.g., SVM, XGBoost)
- Custom preprocessing code beyond the 4 options listed
