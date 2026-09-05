# Diabetes Risk Prediction from Routine Health Metrics

A machine learning pipeline evaluating whether routine, low-cost health metrics — age, BMI, blood pressure, heart rate, cholesterol, temperature, gender, blood type, and smoking status — can predict diabetes status.

**Course:** CSE422 — Artificial Intelligence · **Group 07**
**Authors:** Tahamidul Alam Chowdhury (22299066) · Mehedi Zaman Sami (22299481)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![pandas](https://img.shields.io/badge/pandas-3.0-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E)
![License](https://img.shields.io/badge/license-Academic%20Use-lightgrey)

---

## Table of Contents
- [Overview](#overview)
- [Key Finding](#key-finding)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Report](#report)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

Diabetes is one of the most common chronic conditions worldwide, and a large share of cases go undiagnosed because comprehensive diagnostic testing (fasting glucose, HbA1c) isn't always accessible. This project investigates whether routine, cheaply-collected vitals can serve as a proxy — building and rigorously evaluating five supervised classifiers and an unsupervised clustering model on a 200,000-row health-metrics dataset.

The project follows a complete ML pipeline: exploratory data analysis → pre-processing → model training → multi-metric evaluation → 5-fold cross-validation → conclusion.

## Key Finding

> **The models don't predict diabetes well — and the analysis shows why that's a data limitation, not a pipeline failure.** Every numeric feature's correlation with the target came out under 0.007 (essentially zero) across three independent correlation methods. This was independently confirmed by K-Means clustering (which never saw the labels), a cross-target test (swapping the target for Smoking/Gender gave the same result), and 5-fold cross-validation (confirming the weak results are stable, not a fluke of one split). Full reasoning in the [report](#report).

## Dataset

`7.csv` — 200,000 rows, 13 columns (12 predictor features + 1 target).

| Column | Type | Description |
|---|---|---|
| Student ID | numeric | Row identifier (not a health measurement) |
| Age | numeric | Age in years |
| Gender | categorical | Male / Female |
| Height | numeric | Height (cm) |
| Weight | numeric | Weight (kg) |
| Blood Type | categorical | A / B / AB / O |
| BMI | numeric | Body Mass Index |
| Temperature | numeric | Body temperature (°F) |
| Heart Rate | numeric | Resting heart rate (bpm) |
| Blood Pressure | numeric | Systolic blood pressure |
| Cholesterol | numeric | Cholesterol level |
| **Diabetes** | categorical | **Target** — Yes / No (~90% / 10%) |
| Smoking | categorical | Yes / No |

## Methodology

1. **Exploratory Data Analysis** — distributions, skewness, outlier checks, missing-value patterns (~10% per column), correlation analysis (Pearson/Spearman/Kendall), class-imbalance check.
2. **Pre-processing** — dropped rows with a missing target; removed 7,549 exact duplicates; dropped `Student ID` (not a real identifier) and `Height`/`Weight` (redundant with `BMI`); stratified 80/20 train-test split; median/most-frequent imputation and standard scaling fit on training data only; one-hot encoding for `Blood Type`.
3. **Model Training** — 5 supervised classifiers + a majority-class baseline; K-Means clustering on the same features with labels withheld.
4. **Evaluation** — confusion matrices, accuracy, precision, recall, F1, ROC-AUC for every model; ARI/NMI/silhouette score for K-Means against true labels.
5. **Cross-Validation** — 5-fold Stratified K-Fold re-evaluation of all models, confirming result stability.

## Models Evaluated

| Model | Key configuration |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Gaussian Naive Bayes | default |
| Decision Tree | `max_depth=8`, `class_weight='balanced'` |
| K-Nearest Neighbors | `n_neighbors=15`, distance-weighted |
| Neural Network (MLP) | 2 hidden layers (64, 32), early stopping |
| Baseline | majority-class dummy classifier |
| K-Means (unsupervised) | `k=2`, selected via elbow method |

## Results

Test-set performance, sorted by AUC:

| Model | Accuracy | Precision | Recall | F1 | AUC | CV Mean AUC (5-fold) |
|---|---|---|---|---|---|---|
| **K-Nearest Neighbors** | 0.9115 | 0.9880 | 0.1189 | 0.2122 | **0.6555** | 0.6504 ± 0.0043 |
| Decision Tree | 0.5337 | 0.1049 | 0.4851 | 0.1726 | 0.5215 | 0.5204 ± 0.0036 |
| Gaussian Naive Bayes | 0.8998 | 0.0000 | 0.0000 | 0.0000 | 0.5078 | 0.5019 ± 0.0045 |
| Logistic Regression | 0.5042 | 0.1015 | 0.5025 | 0.1689 | 0.5072 | 0.5025 ± 0.0051 |
| Neural Network (MLP) | 0.8998 | 0.0000 | 0.0000 | 0.0000 | 0.5050 | 0.5039 ± 0.0053 |
| Baseline (majority class) | 0.8998 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 ± 0.0000 |

**K-Means vs. true labels:** Adjusted Rand Index ≈ 0.0000, Normalized Mutual Information ≈ 0.0000, silhouette score 0.096 — no meaningful correspondence between discovered clusters and diabetes status.

## Repository Structure

```
.
├── README.md                              # This file
├── 7.csv                                  # Dataset (200,000 rows × 13 columns)
├── CSE422_Group07_Diabetes_Project.ipynb  # Full analysis notebook (EDA → modelling → evaluation)
├── CSE422_Group07_Report.pdf              # IEEE-format written report
├── CSE422_Group07_Report.tex              # LaTeX source for the report
└── requirements.txt                       # Python dependencies
```

## Getting Started

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook CSE422_Group07_Diabetes_Project.ipynb
```

Then run **Kernel → Restart & Run All**. A full run takes approximately 4–5 minutes (the cross-validation section trains all 6 models across 5 folds each).

> Alternatively, open the notebook directly in [Google Colab](https://colab.research.google.com) — upload the `.ipynb` file and `7.csv` into the same session, then **Runtime → Run all**.

## Report

The full write-up — including the complete correlation analysis, all figures, and detailed discussion of probable causes behind the results — is available as [`CSE422_Group07_Report.pdf`](CSE422_Group07_Report.pdf), formatted per IEEE journal conventions.

## Limitations & Future Work

- No hyperparameter tuning was performed (e.g. via `GridSearchCV`) — cross-validation here validates the *stability* of fixed settings, not the *optimality* of them.
- `class_weight` balancing was only available for 2 of the 5 classifiers in scikit-learn's implementation, so imbalance handling isn't fully consistent across models.
- No formal statistical test (e.g. Chi-Square) was run for categorical-feature independence, only group-rate comparisons.
- The dataset's vitals-only feature set lacks the clinical markers (fasting glucose, HbA1c, family history) that would likely be necessary for a viable real-world screening model.

## Acknowledgments

Completed as coursework for **CSE422 (Artificial Intelligence)**. Shared for educational and portfolio purposes.
