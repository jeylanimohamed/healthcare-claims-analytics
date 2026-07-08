# Healthcare Claims Analytics (Synthetic Data)

**Mohamed Jeylani**

This is a personal healthcare-analytics project built entirely on synthetic claims-style data. I use it to practise the end-to-end flow: generating relational data, querying it with SQL, summarising cost and utilisation, plotting the results, and training a model to classify admissions with a proper patient-level split. I wrote the generator and encoded the relationships in it myself, so the results below show the pipeline working rather than anything true about real patients. There's no protected health information here, no real records, and no employer data.

> My internships gave me experience with analytics, Power BI and executive reporting in healthcare settings. I built this separately, on my own time, to practise claims-style data modelling, SQL, Python and ML on fully synthetic data. It isn't internship work, it doesn't use any employer or claims data, and it isn't a reproduction of anything I did professionally.

## The question I was answering

If I were a claims analyst, I'd want to know how cost and admissions move with diagnosis, region and patient risk tier, and whether a model can pick out the admission flag. I'm answering that inside a dataset I generated, purely to exercise the methods.

## The data (synthetic)

Everything comes from `data/generate_data.py`, using the kinds of fields you'd see in claims data. The diagnosis and procedure combinations are made up for the exercise — none of it is calibrated against a real CMS file, a Medicare population, or any real coding distribution. Ages start at 18, so this is not a Medicare-representative population.

| Table | Rows | What's in it |
|---|---|---|
| `patients` | 5,000 | demographics, chronic-condition flags, a generated risk score |
| `providers` | 200 | specialty, region, panel size |
| `claims` | 25,000 | diagnosis/procedure codes, cost, admission flag, length of stay |

You can check the counts by running the project (below).

To be clear about scope: the clinical relationships are generated, not validated, there's no PHI, and none of the output should be used for clinical, actuarial, reimbursement, policy or care decisions.

## Repository structure

```
healthcare-claims-analytics/
├── README.md
├── requirements.txt
├── analysis.py              # SQL analysis, cost/utilisation summaries, charts, admission classifier
├── data/
│   └── generate_data.py     # synthetic data generator (CSV output is git-ignored)
└── sql/
    ├── queries.sql          # 7 analytical SQL queries
    └── load_to_sqlite.py    # builds data/healthcare.db so the SQL runs on its own
```

## How I approached it

1. **Generate** the patients, providers and claims with a fixed seed. The generator uses age and risk score when it produces cost, admission probability and the risk figure — that structure is deliberate.
2. **SQL analysis** in SQLite: cost by diagnosis, monthly trend, top-cost patients, provider-specialty summary, cost by risk tier, a repeat-admission proxy, and a regional comparison.
3. **Cost and utilisation summaries** in pandas by risk tier, age band and region.
4. **Admission classification** on the generated admission flag: a Random Forest with a patient-level train/test split (`test_size=0.25`, `random_state=42`) so no patient's claims land in both train and test, plus a scaled logistic-regression baseline for comparison.

Everything below describes the generated sample only.

## Results (on the generated data)

The analysis recovers the age/risk/cost/admission structure I built into the generator — that's pipeline validation, not a finding about real patients.

**Cost and utilisation (generated)**

- Total claims cost: $2,438,467.72.
- Cost per patient by risk tier: Low (<1.5) $389.90, Medium $512.75, High (2.5–3.5) $620.56, Very High (>3.5) $851.30.
- Admission-flag rate: 38.7% of claims.
- Repeat-admission proxy: 67.3% of admitted patients had two or more admission-associated claims in the generated year. It's a rough proxy, not a 30-day, unplanned, or CMS readmission measure.

**Admission classifier (synthetic held-out set, patient-level split)**

- Random Forest ROC-AUC 0.720; logistic-regression baseline 0.723.
- Baseline accuracy (always predict "not admitted") 0.609; model accuracy 0.689.
- Admitted class: precision 0.63, recall 0.49. Not-admitted: precision 0.71, recall 0.82.
- Feature importance (RF): `patient_age` 0.657, `patient_risk_score` 0.164. Since the admission label is generated from age and risk, this is the model recovering known structure, not a clinical discovery.

## Limitations

- All synthetic — no real, employer, CMS or patient data, and nothing here is clinical or financial evidence.
- Not Medicare-representative: ages start at 18 and no CMS structure is reproduced.
- Age, risk, cost and admission are linked by the generator, so the model's performance and feature importance just reflect that.
- The repeat-admission proxy (two or more admission-associated claims in a year) isn't aligned to any clinical or CMS readmission definition.
- Metrics are computed on generated labels, so they say nothing about how this would generalise to real claims.

## Running it from a clean clone

```bash
pip install -r requirements.txt

python data/generate_data.py     # writes data/*.csv (git-ignored)
python analysis.py               # SQL analysis, summaries, charts -> output/, admission classifier

# optional: run the SQL file on its own
python sql/load_to_sqlite.py     # builds data/healthcare.db
# then, if you have the sqlite3 CLI:
# sqlite3 data/healthcare.db < sql/queries.sql
```

A normal run reports 5,000 patients, 200 providers, 25,000 claims, a 38.7% admission-flag rate, a 67.3% repeat-admission proxy, and admission ROC-AUC 0.720 (RF) / 0.723 (LR baseline).

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, SQL, SQLite

## Things I'd do next

- Add a properly documented synthetic readmission window (index admission, chronological ordering, 30-day gap), clearly labelled as synthetic and not tied to a specific CMS measure.
- Add calibration and precision-recall curves for the classifier.
- Add explicit data-quality checks to the generation step.

## One-line summary

Personal healthcare-analytics project on fully synthetic claims data (5,000 patients, 200 providers, 25,000 claims): SQL analysis in SQLite, cost/utilisation summaries, charts, and an admission classifier with a patient-level split (ROC-AUC 0.72; logistic-regression baseline 0.72). The relationships are generated, so the results show the pipeline, not real clinical patterns.
