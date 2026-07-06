"""
Synthetic Healthcare Claims Data Generator
Generates realistic Medicare-style claims data for analytics portfolio project.

Schema mirrors real CMS patterns:
- Claims: procedure codes, diagnosis codes, costs, dates
- Patients: demographics, chronic conditions, risk scores
- Providers: specialty, region, patient panel size
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_PATIENTS = 5000
N_PROVIDERS = 200
N_CLAIMS = 25000

# --- Patients ---
def generate_patients(n):
    ages = np.random.normal(62, 14, n).clip(18, 95).astype(int)
    genders = np.random.choice(['M', 'F'], n, p=[0.48, 0.52])
    regions = np.random.choice(['Midwest', 'Northeast', 'South', 'West', 'Pacific'], n, p=[0.22, 0.18, 0.35, 0.15, 0.10])

    conditions = ['Diabetes', 'Hypertension', 'CAD', 'COPD', 'CKD', 'CHF', 'Asthma', 'Depression']
    condition_weights = {c: np.random.beta(2, 8) + 0.05 * (i + 1) for i, c in enumerate(conditions)}
    n_conditions = np.random.poisson(1.8, n).clip(0, len(conditions))

    chronic_conditions = []
    risk_scores = []
    for i in range(n):
        nc = min(n_conditions[i], len(conditions))
        probs = np.array([condition_weights[c] for c in conditions])
        probs = probs / probs.sum()
        selected = list(np.random.choice(conditions, nc, replace=False, p=probs))
        chronic_conditions.append('; '.join(selected) if selected else 'None')

        risk = 1.0 + 0.3 * len(selected) + 0.02 * max(0, ages[i] - 50) + np.random.normal(0, 0.5)
        risk_scores.append(round(max(0.2, risk), 2))

    return pd.DataFrame({
        'patient_id': [f'P{str(i).zfill(5)}' for i in range(1, n + 1)],
        'age': ages,
        'gender': genders,
        'region': regions,
        'chronic_conditions': chronic_conditions,
        'num_chronic_conditions': n_conditions,
        'risk_score': risk_scores
    })

# --- Providers ---
def generate_providers(n):
    specialties = ['PCP', 'Cardiology', 'Endocrinology', 'Pulmonology', 'Nephrology', 'Orthopedics', 'Oncology', 'Neurology']
    regions = ['Midwest', 'Northeast', 'South', 'West', 'Pacific']
    weights = [0.30, 0.12, 0.08, 0.06, 0.06, 0.10, 0.08, 0.06]
    weights_region = [0.22, 0.18, 0.35, 0.15, 0.10]

    return pd.DataFrame({
        'provider_id': [f'PRV{str(i).zfill(4)}' for i in range(1, n + 1)],
        'specialty': np.random.choice(specialties, n, p=np.array(weights) / sum(weights)),
        'region': np.random.choice(regions, n, p=np.array(weights_region) / sum(weights_region)),
        'panel_size': np.random.randint(200, 3000, n)
    })

# --- Claims ---
def generate_claims(patients, providers, n):
    procedure_codes = ['99213', '99214', '99215', '93000', '80053', '84443', '97110', 'G0439', 'G0444', '36415']
    proc_costs = {'99213': 73, '99214': 110, '99215': 148, '93000': 24, '80053': 35, '84443': 18, '97110': 31, 'G0439': 172, 'G0444': 26, '36415': 12}
    dx_codes = ['E11.9', 'I10', 'I25.10', 'J44.9', 'N18.3', 'I50.9', 'J45.909', 'F32.9']
    dx_descriptions = {
        'E11.9': 'Diabetes type 2', 'I10': 'Hypertension', 'I25.10': 'CAD',
        'J44.9': 'COPD', 'N18.3': 'CKD stage 3', 'I50.9': 'Heart failure',
        'J45.909': 'Asthma', 'F32.9': 'Depression'
    }

    p_ids = patients['patient_id'].values
    prv_ids = providers['provider_id'].values
    p_ages = dict(zip(patients['patient_id'], patients['age']))
    p_regions = dict(zip(patients['patient_id'], patients['region']))
    p_risk = dict(zip(patients['patient_id'], patients['risk_score']))
    prv_spec = dict(zip(providers['provider_id'], providers['specialty']))
    prv_region = dict(zip(providers['provider_id'], providers['region']))

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    days_range = (end_date - start_date).days

    records = []
    for _ in range(n):
        pid = np.random.choice(p_ids)
        age = p_ages[pid]

        # Older / sicker patients have more claims (they're picked more often)
        dx = np.random.choice(dx_codes)
        proc = np.random.choice(procedure_codes)
        base_cost = proc_costs[proc]
        age_factor = 1.0 + max(0, (age - 50) * 0.01)
        risk_factor = 0.7 + 0.3 * p_risk[pid]
        cost = base_cost * age_factor * risk_factor * np.random.uniform(0.8, 1.3)

        service_date = start_date + timedelta(days=np.random.randint(0, days_range))

        # Pick provider in same region 70% of the time
        p_region_val = p_regions[pid]
        same_region_provs = providers[providers['region'] == p_region_val]['provider_id'].values
        if len(same_region_provs) > 0 and np.random.random() < 0.7:
            provider = np.random.choice(same_region_provs)
        else:
            provider = np.random.choice(prv_ids)

        # Admission flag (higher for older / multi-condition patients)
        admitted = 1 if (np.random.random() < 0.12 + 0.015 * max(0, age - 50) + 0.04 * p_risk[pid]) else 0
        los = max(1, int(np.random.exponential(3.5))) if admitted else 0

        records.append({
            'claim_id': f'CLM{str(len(records) + 1).zfill(6)}',
            'patient_id': pid,
            'provider_id': provider,
            'service_date': service_date.strftime('%Y-%m-%d'),
            'diagnosis_code': dx,
            'diagnosis_desc': dx_descriptions[dx],
            'procedure_code': proc,
            'total_cost': round(cost, 2),
            'admitted': admitted,
            'length_of_stay': los,
            'patient_age': age,
            'patient_region': p_region_val,
            'patient_risk_score': p_risk[pid],
            'provider_specialty': prv_spec[provider],
            'provider_region': prv_region[provider]
        })

    return pd.DataFrame(records).sort_values('service_date').reset_index(drop=True)

if __name__ == '__main__':
    patients = generate_patients(N_PATIENTS)
    providers = generate_providers(N_PROVIDERS)
    claims = generate_claims(patients, providers, N_CLAIMS)

    patients.to_csv('data/patients.csv', index=False)
    providers.to_csv('data/providers.csv', index=False)
    claims.to_csv('data/claims.csv', index=False)

    print(f"Generated {len(patients)} patients, {len(providers)} providers, {len(claims)} claims.")
    print(f"Files saved to data/patients.csv, data/providers.csv, data/claims.csv")
