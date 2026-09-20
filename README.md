# Ethical AI — Fairness & Explainability Analysis

A machine learning project that trains an income-prediction model and analyzes it for **fairness** and **explainability**, using the UCI Adult Income dataset. Built for the DAAI Lesson 14 (Ethical Analysis and Explainability) assignment.

---

## 📌 Overview

This project builds a logistic regression model to predict whether a person earns more than $50K a year, then goes beyond accuracy to ask two ethical questions:

- **Fairness:** Does the model treat men and women equally?
- **Explainability:** Can we understand *why* the model makes its decisions?

Using Fairlearn, SHAP, and LIME, the project shows that a model can be reasonably accurate yet still biased — and demonstrates how to detect and explain that bias.

---

## 🗂️ Repository contents

| File | Description |
|------|-------------|
| `Ethical_Analysis_Fairness_Explainability.ipynb` | Main Colab notebook — full pipeline |
| `fairness_metrics.png` | Fairness metrics by sex (bar plots) |
| `shap_global.png` | SHAP global feature importance |
| `shap_local.png` | SHAP waterfall for a single prediction |
| `fairness_model.pkl` | The saved trained model |
| `README.md` | This file |

---

## 🚀 Setup & how to run

Developed in **Google Colab** (a standard CPU runtime is fine — no GPU needed).

1. Open [Google Colab](https://colab.research.google.com/) and upload `Ethical_Analysis_Fairness_Explainability.ipynb`.
2. Run the cells in order (**Runtime → Run all**).

The first cell installs the required libraries:

```bash
pip install fairlearn shap lime
```

(`pandas`, `numpy`, `scikit-learn`, and `matplotlib` come pre-installed in Colab.)

---

## 🔬 Pipeline

1. **Setup** — install and import libraries
2. **Data preparation** — load the Adult dataset, clean missing values (`?`), build features (X), target (y), and the sensitive attribute (sex)
3. **Model training** — 80/20 train/test split, feature scaling, logistic regression
4. **Evaluation** — accuracy, confusion matrix, classification report
5. **Fairness analysis** — Fairlearn MetricFrame + bar plots by sex
6. **Explainability** — SHAP (global bar + local waterfall) and LIME (local)
7. **Save model** — export the trained model

---

## 📊 Dataset

**UCI Adult Income dataset** (~30,000 records after cleaning) — predicts whether income exceeds $50K/year from census features such as age, education, occupation, hours worked, and marital status.

- **Target:** income (>50K = 1, ≤50K = 0)
- **Sensitive attribute:** sex (used for fairness analysis)
- **Note:** missing values are stored as `?` and are cleaned during preprocessing.

Source: https://archive.ics.uci.edu/ml/datasets/adult

---

## 📈 Key results

| Area | Finding |
|------|---------|
| **Accuracy** | ~85% overall, but weaker recall (61%) on the minority >50K class due to class imbalance |
| **Fairness** | Selection rate: **26% for men vs 9% for women** — men are far more likely to be predicted as high earners |
| **Explainability** | SHAP shows **sex is a top-ranked feature**, confirming the model learned gender bias from historical data; LIME agrees with SHAP on the main drivers |

**Takeaway:** the model inherits bias present in historical data. Before any real-world use (lending, hiring), bias-mitigation techniques and human oversight would be required.

---

## 🛠️ Tech stack

- **Python 3**
- **scikit-learn** — logistic regression, preprocessing, metrics
- **Fairlearn** — fairness metrics (MetricFrame)
- **SHAP** — global and local explainability
- **LIME** — local explainability
- **pandas / numpy / matplotlib** — data handling and plots

---

## 📄 License

Educational project. The UCI Adult dataset is publicly available for research use.

---

## ✍️ Author

**Hiren Patel** — Data Analytics & AI (DAAI)
