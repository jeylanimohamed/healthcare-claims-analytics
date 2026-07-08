# Healthcare Claims Analytics (Synthetic Portfolio Project)

**Mohamed Jeylani**

This is an independent healthcare-analytics portfolio project built on **fully synthetic claims-style data**. It demonstrates an end-to-end workflow — synthetic relational data generation, SQL analysis via SQLite, cost and utilisation summaries, visualisation, and an admission-classification model with a patient-level train/test split. Several relationships are intentionally encoded in the generator, so the reported findings **demonstrate the analytical pipeline rather than real clinical or financial patterns**. The data contain **no protected health information**, no real patient records, and no employer data.

> My internships developed my experience in analytics, Power BI and executive reporting within healthcare organisations. This separate personal project was created to practise claims-style data modelling, SQL, Python and machine-learning methods using **entirely synthetic data**. It is not internship work, does not use employer or claims data, and is not a reproduction of anything I did professionally.

---

## Business question (illustrative)

Framed as a claims analyst might: within a generated dataset, how do cost and admission utilisation vary by diagnosis, region and patient risk tier, and can a model classify the (generated) admission flag? These are answered **only within synthetic data**, to exercise the methods.

---

## Dataset (synthetic)

All data are generated locally by `data/generate_data.py`, using common healthcare fields. The diagnosis/procedure combinations and clinical relationships are **generated for demonstration** and are **not** calibrated to or validated against any real CMS file, Medicare population, or coding distribution. Patient ages range from 18 upward, so this is **not** a Medicare/CMS-representative population.

| Table | Rows | Notes |
|---|---|---|
| `patients` | 5,000 | demographics, chronic-condition flags, a generated risk score |
| `providers` | 200 | specialty, region, panel size |
| `claims` | 25,000 | diagnosis/procedure codes, cost, admission flag, length of stay |

Verify counts by running the project (see below).

**Disclaimer.** Diagnosis and procedure combinations are generated; clinical relationships are not validated; the data contain no PHI; outputs are **not** suitable for clinical, actuarial, reimbursement, policy, or care-management decisions.

---

## Repository structure

```
healthcare-claims-analytics/
├── README.md
├── requirements.txt
├── analysis.py              # SQL analysis, cost/utilisation summaries, charts, admission classifier
├── data/
│   └── generate_data.py     # synthetic data generator (git-ignored CSV output)
└── sql/
    ├── queries.sql          # 7 analytical SQL queries
    └── load_to_sqlite.py    # builds data/healthcare.db so the SQL runs standalone
```

---

## Methodology

1. **Generate** synthetic patients, providers and claims (fixed seed). The generator uses age and risk score when producing cost, admission probability and risk — an intentional, encoded structure.
2. **SQL analysis** via SQLite: cost by diagnosis, monthly trend, top-cost patients, provider-specialty summary, cost by risk tier, repeat-admission proxy, regional comparison.
3. **Cost and utilisation summaries** in pandas by risk tier, age band and region.
4. **Admission classification** on the generated admission flag: Random Forest with a **patient-level** train/test split (`test_size=0.25`, `random_state=42`) so a patient's claims cannot appear in both train and test. A scaled **logistic-regression baseline** is reported for comparison.

Reported outputs describe **the generated sample only**.

---

## Results (outputs of the generated sample)

The analysis recovers the age/risk/cost/admission relationships intentionally encoded in the generator; this is pipeline validation, **not** evidence about real patient populations.

**Cost and utilisation (generated)**
- Total generated claims cost: **$2,438,467.72**.
- Cost per patient by risk tier: Low (<1.5) **$389.90**, Medium **$512.75**, High (2.5–3.5) **$620.56**, Very High (>3.5) **$851.30**.
- Admission-flag rate: **38.7%** of generated claims.
- **Repeat-admission proxy: 67.3%** of admitted patients had ≥2 admission-associated claims in the generated year. This is a proxy, **not** a 30-day, unplanned, or CMS readmission measure.

**Admission classification (synthetic held-out test set, patient-level split)**
- Random Forest ROC-AUC **0.720**; logistic-regression baseline ROC-AUC **0.723**.
- Baseline accuracy (predict "not admitted") 0.609; model accuracy 0.689.
- Admitted class: precision 0.63, recall 0.49. Not-admitted: precision 0.71, recall 0.82.
- Feature importance (RF): `patient_age` 0.657, `patient_risk_score` 0.164. Because the admission label is generated from age and risk, this reflects **recovery of encoded structure**, not a clinical discovery.

---

## Limitations

- **Synthetic data.** No real, employer, CMS, or patient data; outputs are not clinical, actuarial, or financial evidence.
- **Not Medicare/CMS-representative.** Ages start at 18 and no CMS structure is reproduced or validated.
- **Encoded relationships.** Age, risk, cost and admission are linked by the generator, so model performance and feature importance recover known structure.
- **Repeat-admission proxy.** Defined as ≥2 admission-associated claims in the year; not aligned to any clinical/CMS readmission definition.
- **Evaluation.** Metrics are on generated labels, so they do not indicate generalisation to real claims or clinical populations.

---

## Reproduce from a clean clone

```bash
pip install -r requirements.txt

python data/generate_data.py     # writes data/*.csv (git-ignored)
python analysis.py               # SQL analysis, summaries, charts -> output/, admission classifier

# Optional: run the SQL file standalone
python sql/load_to_sqlite.py     # builds data/healthcare.db
# then, if the sqlite3 CLI is installed:
# sqlite3 data/healthcare.db < sql/queries.sql
```

A typical run reports 5,000 patients, 200 providers, 25,000 claims, admission-flag rate 38.7%, repeat-admission proxy 67.3%, and admission-classification ROC-AUC 0.720 (RF) / 0.723 (LR baseline).

---

## Technologies

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, SQL, SQLite

---

## Potential next steps

- Implement a documented synthetic readmission window (index admission, chronological ordering, 30-day gap) — clearly labelled synthetic and not aligned to a specific CMS measure unless verified.
- Add calibration and precision-recall curves for the admission classifier.
- Add explicit data-quality checks to the generation step.

---

## Résumé summary

Independent healthcare-analytics project on **fully synthetic** claims-style data (5,000 patients, 200 providers, 25,000 claims): SQL analysis via SQLite, cost/utilisation summaries, visualisation, and an admission-classification model with a **patient-level** train/test split (ROC-AUC 0.72; logistic-regression baseline 0.72). Relationships are generator-encoded, so results demonstrate the pipeline, not real clinical patterns.
