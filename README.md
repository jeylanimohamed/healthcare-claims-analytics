# Healthcare Claims Analytics Project

**Portfolio project — Mohamed Jeylani** | July 2026

Analysis of 25,000 synthetic Medicare-style healthcare claims across 5,000 patients and 200 providers. Built to demonstrate Python, SQL, and predictive modelling skills for data analyst roles.

---

## Project Overview

This project simulates a real-world healthcare analytics workflow — the kind of work done by data analysts at UnitedHealth Group, Optum, and other payers. It covers:

- **Data generation** — realistic claims data with patient demographics, diagnoses, procedures, costs, and admissions
- **SQL analysis** — 7 analytical queries answering business questions about cost drivers, regional variation, and readmissions
- **Python analysis** — pandas, matplotlib, and seaborn for exploratory data analysis and visualisation
- **Predictive modelling** — Random Forest classifier to predict hospital admission risk (ROC-AUC: 0.72)

---

## Dataset

| Table | Records | Description |
|---|---|---|
| `patients` | 5,000 | Demographics, chronic conditions, risk scores |
| `providers` | 200 | Specialty, region, patient panel size |
| `claims` | 25,000 | Diagnosis codes, procedure costs, admission flags, length of stay |

Generated via `data/generate_data.py` — mirrors CMS claims data patterns.

---

## Key Findings

1. **High-risk patients cost 1.6× more** than low-risk ($621 vs $390 per patient) — actionable for care management programmes
2. **67.3% of admitted patients** are readmitted within the year — significant opportunity for intervention programmes
3. **Age is the dominant predictor** of admission risk (67% feature importance), followed by risk score (16%)
4. **38.7% of claims** result in hospital admission, driving 41.5% of total costs

---

## Files

```
healthcare-analytics-project/
├── README.md
├── requirements.txt
├── analysis.py              # Main analysis + visualisations + ML model
├── data/
│   ├── generate_data.py     # Synthetic claims data generator
│   ├── patients.csv
│   ├── providers.csv
│   └── claims.csv
├── sql/
│   └── queries.sql          # 7 analytical SQL queries
├── output/                  # Generated charts (see below)
│   ├── diagnosis_cost_analysis.png
│   ├── monthly_trend.png
│   ├── risk_score_impact.png
│   ├── regional_comparison.png
│   └── feature_importance.png
└── notebooks/               # For Jupyter exploration (optional)
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate the data (or skip — data is pre-generated in data/)
python3 data/generate_data.py

# Run the full analysis
python3 analysis.py
```

---

## SQL Queries Included

1. Total cost by diagnosis
2. Monthly cost trend
3. Top 10 highest-cost patients
4. Provider specialty performance
5. High-risk patient cost analysis
6. Readmission analysis
7. Regional cost comparison

---

## Technologies

`Python` · `pandas` · `NumPy` · `matplotlib` · `seaborn` · `scikit-learn` · `SQLite` · `SQL`

---

## Context

Built as part of a job search targeting data analyst and business intelligence roles in London (UK) and the Netherlands. This project connects directly to experience at UnitedHealth Group and Optum, where I worked with claims data, executive dashboards, and population health analytics.
