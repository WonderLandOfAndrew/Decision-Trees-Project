# 🌳 C4.5 Decision Tree — Gym Session Duration Classifier

A from-scratch implementation of the **C4.5 decision tree algorithm** applied to a gym members dataset. The model classifies workout session duration into three categories — **Short**, **Medium**, and **Long** — achieving **97.3% accuracy** on the held-out test set.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Notebook Walkthrough](#notebook-walkthrough)
- [C4.5 Algorithm — Key Concepts](#c45-algorithm--key-concepts)
- [Feature Engineering](#feature-engineering)
- [Results](#results)
- [Decision Tree Visualization](#decision-tree-visualization)

---

## Overview

This project builds a **C4.5-style decision tree classifier** from scratch using Python, without relying on any black-box tree library for the core logic. It follows the same general pipeline as a standard ML workflow:

1. Load & inspect data
2. Engineer domain-aware features
3. Encode the target variable (continuous → 3 classes)
4. Prepare train / validation / test splits
5. Balance the training set with SMOTE
6. Apply Robust scaling
7. Train the tree using gain ratio splits
8. Prune with validation data
9. Tune hyperparameters via grid search
10. Evaluate on the final test set

> **Why C4.5?** C4.5 extends the classic ID3 algorithm by supporting continuous features, using **gain ratio** (instead of raw information gain) to avoid bias toward high-cardinality features, and adding **reduced-error pruning** to improve generalization.

---

## Dataset

**File:** `gym_members_exercise_tracking.csv`

| Property | Value |
|---|---|
| Rows | 973 |
| Features | 15 (original) + 7 engineered = 22 total |
| Missing values | None |
| Target | `Session_Duration (hours)` → binned into 3 classes |

**Original columns:**

`Age`, `Gender`, `Weight (kg)`, `Height (m)`, `Max_BPM`, `Avg_BPM`, `Resting_BPM`, `Session_Duration (hours)`, `Calories_Burned`, `Workout_Type`, `Fat_Percentage`, `Water_Intake (liters)`, `Workout_Frequency (days/week)`, `Experience_Level`, `BMI`

**Target classes (quantile-based binning):**

| Class | Session Duration |
|---|---|
| Short | ≤ 33rd percentile |
| Medium | 33rd – 67th percentile |
| Long | ≥ 67th percentile |

---

## Project Structure

```
.
├── c45_decision_tree_sportsman.ipynb   # Main notebook
├── gym_members_exercise_tracking.csv  # Dataset
├── Digraph.gv                         # Graphviz source for the decision tree
├── Digraph_gv.pdf                     # Rendered decision tree (PDF)
└── README.md
```

---

## Requirements

Install all dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```

| Package | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` | Plots and confusion matrix |
| `scikit-learn` | Metrics, train/test split, scaler, parameter grid |
| `imbalanced-learn` | SMOTE oversampling |

> Python 3.8+ is recommended. The notebook was developed on Python 3.11.

---

## How to Run

1. Clone or download this repository.
2. Place `gym_members_exercise_tracking.csv` in the same folder as the notebook **or** at `/mnt/data/gym_members_exercise_tracking.csv`.
3. Launch Jupyter:

```bash
jupyter notebook c45_decision_tree_sportsman.ipynb
```

4. Run all cells top-to-bottom (**Kernel → Restart & Run All**).

---

## Notebook Walkthrough

The notebook is organized into clearly labeled sections:

### 1 — Imports
Standard library imports: `pandas`, `numpy`, `matplotlib`, `sklearn` metrics and model selection utilities.

### 2 — Load Data
Loads the CSV from the local path or `/mnt/data/` fallback. Prints the first 5 rows so you can confirm the data loaded correctly.

### 3 — Inspect Dataset
Prints shape `(973, 15)`, column names, missing value counts, and dtypes. All columns are complete — no imputation needed.

### 4 — Descriptive Statistics
Full `.describe()` including categorical columns. Useful for spotting outliers and understanding feature ranges before engineering.

### 5 — Target Encoding
Converts the continuous `Session_Duration (hours)` into a 3-class target `Session_Class` using dataset quantiles:

```python
Short   →  duration ≤ Q33
Medium  →  Q33 < duration ≤ Q67
Long    →  duration > Q67
```

### 6 — Feature Engineering
Seven new interaction features are created to make class-discriminating patterns explicit (see [Feature Engineering](#feature-engineering) below).

### 7 — Feature / Target Split
Drops the original regression target and the intermediate `Session_Class` column; builds `X` (22 features) and `y` (string labels).

### 8 — Train / Validation / Test Split
Stratified split: **70% train**, **15% validation**, **15% test**.

```
Train:      681 samples
Validation: 146 samples
Test:       146 samples
```

### 9 — SMOTE Oversampling
The training set is class-imbalanced (`Medium ≈ 3× Short`). SMOTE generates synthetic minority samples so the tree sees a balanced 411/411/411 distribution. Validation and test sets are **never touched**.

### 10 — Robust Scaling
`RobustScaler` (median/IQR) is fit on the training set and applied to all three splits. This prevents outliers from skewing split thresholds, particularly for the Short/Medium boundary.

### 11 — C4.5 Implementation
The `C45Classifier` class is implemented from scratch. Core components:

- **`entropy()`** — Shannon entropy of a label distribution
- **`information_gain()`** — entropy reduction for a proposed split
- **`gain_ratio()`** — information gain divided by split information (penalizes wide splits)
- **`best_split()`** — iterates over all features and thresholds to find the highest gain-ratio split
- **`fit()`** — recursive tree building with `max_depth`, `min_samples_split`, `min_samples_leaf`, and `min_gain_ratio` stopping criteria
- **`prune()`** — reduced-error pruning using the validation set (replaces subtrees with leaf nodes if accuracy does not decrease)
- **`predict()`** — traverses the tree for each sample

### 12 — Initial Training & Evaluation
Trains with default hyperparameters. Reports validation accuracy and a full classification report on the test set.

### 13 — Confusion Matrix
`ConfusionMatrixDisplay` plot showing per-class prediction quality.

### 14 — Hyperparameter Grid Search
Exhaustive search over:

| Parameter | Values |
|---|---|
| `max_depth` | 12, 15, 20 |
| `min_samples_split` | 2, 4, 6 |
| `min_samples_leaf` | 4, 8, 12 |
| `min_gain_ratio` | 1e-5, 1e-4, 5e-4 |

Best configuration found: `max_depth=12`, `min_samples_split=2`, `min_samples_leaf=4`, `min_gain_ratio=1e-5`.

### 15 — Final Model
Retrained on train + validation combined using the best params. No post-fit pruning (parameters were already tuned on the validation set).

---

## C4.5 Algorithm — Key Concepts

| Concept | Description |
|---|---|
| **Entropy** | Measures impurity of a node: `H = -Σ p·log₂(p)` |
| **Information Gain** | Entropy reduction achieved by a split |
| **Split Information** | Entropy of the split distribution itself |
| **Gain Ratio** | `IG / SplitInfo` — normalizes gain to avoid bias toward features with many values |
| **Continuous features** | Each unique midpoint between sorted values is evaluated as a candidate threshold |
| **Reduced-error pruning** | Post-hoc: replace an internal node with its majority-class leaf if validation accuracy does not drop |

> **C4.5 vs ID3:** ID3 only handles categorical features and uses raw information gain. C4.5 adds continuous splits, gain ratio normalization, and pruning.

---

## Feature Engineering

Seven interaction features were added on top of the 15 original columns:

| Feature | Formula | Intuition |
|---|---|---|
| `HR_Reserve_Ratio` | `(Avg_BPM - Resting_BPM) / (Max_BPM - Resting_BPM)` | Cardiovascular effort relative to capacity |
| `Intensity_Score` | `Avg_BPM × Freq / BMI` | Combined load normalized by body size |
| `Cal_per_BPM` | `Calories_Burned / Avg_BPM` | Caloric efficiency per heartbeat |
| `Cal_x_Experience` | `Calories_Burned × Experience_Level` | Short≈574, Medium≈1522, Long≈3795 (highly separable) |
| `Exp_x_Freq` | `Experience_Level × Workout_Frequency` | Combined activity load |
| `Cal_per_Freq` | `Calories_Burned / Workout_Frequency` | Per-session intensity proxy |
| `FatPct_x_Exp` | `Fat_Percentage × Experience_Level` | Beginners with higher fat% cluster in Short |
| `BPM_Efficiency` | `Resting_BPM / Avg_BPM` | Cardiovascular efficiency ratio |

---

## Results

### Best Model Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Long | 1.00 | 1.00 | 1.00 | 29 |
| Medium | 0.98 | 0.97 | 0.97 | 88 |
| Short | 0.90 | 0.93 | 0.92 | 29 |
| **Accuracy** | | | **0.97** | **146** |
| Macro avg | 0.96 | 0.97 | 0.96 | 146 |
| Weighted avg | 0.97 | 0.97 | 0.97 | 146 |

**Validation accuracy:** 0.9726  
**Test accuracy:** 0.9726  
**Pruning improvement:** 0.9589 → 0.9726 (+1.4%)

### Features Used in the Final Pruned Tree

| Feature | Splits |
|---|---|
| `Experience_Level` | 1 |
| `Cal_per_BPM` | 1 |
| `FatPct_x_Exp` | 1 |

The pruned tree is remarkably compact — just **3 decision nodes / 7 nodes total** — yet achieves near-perfect accuracy.

---

## Decision Tree Visualization

The final pruned tree (from `Digraph.gv`) is shown below:

```
Experience_Level ≤ 2.5?
├── YES → Cal_per_BPM ≤ 4.501?
│         ├── YES → Predict: Long
│         └── NO  → Predict: Short
└── NO  → Predict: Medium
```

The rendered version is available as **`Digraph_gv.pdf`**.

**Reading the tree:**

- A member with `Experience_Level ≤ 2` (beginner/intermediate) who also burns more than ~4.5 calories per BPM is predicted to have a **Long** session.
- A member with `Experience_Level ≤ 2` but lower caloric efficiency is predicted to have a **Short** session.
- A member with `Experience_Level > 2` (advanced) is predicted to have a **Medium** session.

---

*Built as part of a machine learning course assignment exploring decision tree variants (ID3 → C4.5).*
