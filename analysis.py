"""
Healthcare Claims Analytics — Main Analysis
=============================================
Portfolio project: analyses 25,000 synthetic Medicare-style claims
across 5,000 patients and 200 providers.

Sections:
  1. Data loading + SQL queries via SQLite
  2. Cost analysis by diagnosis, region, and risk tier
  3. Admission & readmission patterns
  4. Visualisation: cost trends, risk heatmaps, regional comparisons
  5. Predictive model: admission risk classifier
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

sns.set_style('whitegrid')
sns.set_palette('mako')

# ── 1. Load Data ───────────────────────────────────────────
patients = pd.read_csv('data/patients.csv')
providers = pd.read_csv('data/providers.csv')
claims = pd.read_csv('data/claims.csv')

claims['service_date'] = pd.to_datetime(claims['service_date'])
claims['month'] = claims['service_date'].dt.to_period('M')

print(f"Loaded: {len(patients):,} patients | {len(providers):,} providers | {len(claims):,} claims")
print(f"Date range: {claims['service_date'].min().date()} to {claims['service_date'].max().date()}")
print(f"Total cost: ${claims['total_cost'].sum():,.2f}\n")

# ── 1b. SQL Queries via SQLite ─────────────────────────────
conn = sqlite3.connect(':memory:')
patients.to_sql('patients', conn, index=False)
providers.to_sql('providers', conn, index=False)
claims_sql = claims.drop(columns=['month']).copy()
claims_sql.to_sql('claims', conn, index=False)

print("=" * 60)
print("SQL: Top 5 Diagnoses by Total Cost")
print("=" * 60)
q1 = pd.read_sql_query("""
    SELECT diagnosis_desc, COUNT(*) as claims,
           ROUND(AVG(total_cost),2) as avg_cost,
           ROUND(SUM(total_cost),2) as total_cost
    FROM claims
    GROUP BY diagnosis_desc
    ORDER BY total_cost DESC LIMIT 5
""", conn)
print(q1.to_string(index=False))
print()

print("=" * 60)
print("SQL: Regional Cost Comparison")
print("=" * 60)
q2 = pd.read_sql_query("""
    SELECT patient_region,
           COUNT(DISTINCT patient_id) as patients,
           ROUND(SUM(total_cost),2) as total_cost,
           ROUND(AVG(total_cost),2) as avg_per_claim,
           ROUND(SUM(admitted)*100.0/COUNT(*),1) as admit_rate_pct
    FROM claims GROUP BY patient_region ORDER BY total_cost DESC
""", conn)
print(q2.to_string(index=False))
print()

# ── 2. Cost Analysis ─────────────────────────────────────
print("=" * 60)
print("COST ANALYSIS")
print("=" * 60)

cost_by_risk = claims.groupby(
    pd.cut(claims['patient_risk_score'], bins=[0, 1.5, 2.5, 3.5, 10],
           labels=['Low (<1.5)', 'Medium (1.5-2.5)', 'High (2.5-3.5)', 'Very High (>3.5)'])
).agg(
    patient_count=('patient_id', 'nunique'),
    claim_count=('claim_id', 'count'),
    total_cost=('total_cost', 'sum'),
    avg_cost=('total_cost', 'mean'),
    admission_rate=('admitted', 'mean')
).round(2)

cost_by_risk['cost_per_patient'] = (cost_by_risk['total_cost'] / cost_by_risk['patient_count']).round(2)
print(cost_by_risk)
print()

# Monthly trend
monthly = claims.groupby('month').agg(
    claims=('claim_id', 'count'),
    total_cost=('total_cost', 'sum'),
    patients=('patient_id', 'nunique'),
    admissions=('admitted', 'sum')
).reset_index()
monthly['month'] = monthly['month'].astype(str)
print("Monthly trend (first 3 + last 3 months):")
print(monthly.head(3).to_string(index=False))
print("  ...")
print(monthly.tail(3).to_string(index=False))
print()

# ── 3. Admission Analysis ─────────────────────────────────
print("=" * 60)
print("ADMISSION & READMISSION ANALYSIS")
print("=" * 60)

admit_rate = claims['admitted'].mean() * 100
admit_cost = claims[claims['admitted'] == 1]['total_cost'].sum()
total_cost = claims['total_cost'].sum()
print(f"Overall admission rate: {admit_rate:.1f}%")
print(f"Admission costs: ${admit_cost:,.0f} ({admit_cost/total_cost*100:.1f}% of total)")

admit_by_age = claims.groupby(pd.cut(claims['patient_age'], bins=[18, 40, 55, 65, 75, 100],
    labels=['18-40', '41-55', '56-65', '66-75', '76+'])).agg(
    admit_rate=('admitted', 'mean'),
    avg_los=('length_of_stay', 'mean'),
    claim_count=('claim_id', 'count')
).round(3)
admit_by_age['admit_rate_pct'] = (admit_by_age['admit_rate'] * 100).round(1)
print("\nAdmission rate by age group:")
print(admit_by_age[['admit_rate_pct', 'avg_los', 'claim_count']].to_string())
print()

# Readmissions
readmit = claims[claims['admitted'] == 1].groupby('patient_id').filter(lambda x: len(x) >= 2)
readmit_counts = claims[claims['admitted'] == 1].groupby('patient_id').size()
readmitted_patients = (readmit_counts >= 2).sum()
total_admitted_patients = claims[claims['admitted'] == 1]['patient_id'].nunique()
print(f"Patients with ≥2 admissions: {readmitted_patients} "
      f"({readmitted_patients/total_admitted_patients*100:.1f}% of admitted patients)")
print()

# ── 4. Visualisations ─────────────────────────────────────
print("=" * 60)
print("Generating visualisations...")
print("=" * 60)

# Fig 1: Cost by Diagnosis + Admission Rate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dx_costs = claims.groupby('diagnosis_desc').agg(
    total_cost=('total_cost', 'sum'),
    avg_cost=('total_cost', 'mean'),
    admit_rate=('admitted', 'mean')
).sort_values('total_cost', ascending=True)

bars = axes[0].barh(dx_costs.index, dx_costs['total_cost'] / 1e6, color=sns.color_palette('mako', 8))
axes[0].set_xlabel('Total Cost ($M)')
axes[0].set_title('Total Cost by Diagnosis', fontweight='bold')
axes[0].set_xlim(0, dx_costs['total_cost'].max() / 1e6 * 1.15)
for bar, val in zip(bars, dx_costs['total_cost'] / 1e6):
    axes[0].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'${val:.2f}M', va='center', fontsize=8)

axes[1].scatter(dx_costs['avg_cost'], dx_costs['admit_rate'] * 100, s=200, c=sns.color_palette('mako', 8), edgecolors='white', linewidth=1)
# Add margin around scatter points for labels
x_margin = (dx_costs['avg_cost'].max() - dx_costs['avg_cost'].min()) * 0.3
y_margin = (dx_costs['admit_rate'].max() - dx_costs['admit_rate'].min()) * 100 * 0.3
axes[1].set_xlim(dx_costs['avg_cost'].min() - x_margin, dx_costs['avg_cost'].max() + x_margin)
axes[1].set_ylim(dx_costs['admit_rate'].min() * 100 - y_margin, dx_costs['admit_rate'].max() * 100 + y_margin)
for dx, row in dx_costs.iterrows():
    axes[1].annotate(dx, (row['avg_cost'], row['admit_rate'] * 100), fontsize=7, ha='center', va='bottom')
axes[1].set_xlabel('Avg Cost per Claim ($)')
axes[1].set_ylabel('Admission Rate (%)')
axes[1].set_title('Cost vs Admission Rate by Diagnosis', fontweight='bold')

plt.tight_layout()
plt.savefig('output/diagnosis_cost_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ output/diagnosis_cost_analysis.png")

# Fig 2: Monthly Trend
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.fill_between(range(len(monthly)), monthly['total_cost'] / 1e6, alpha=0.3, color='#2c7fb8')
ax1.plot(range(len(monthly)), monthly['total_cost'] / 1e6, 'o-', color='#2c7fb8', linewidth=2, markersize=6)
ax1.set_xlabel('Month (2024)')
ax1.set_ylabel('Total Cost ($M)', color='#2c7fb8')
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly['month'].values, rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(range(len(monthly)), monthly['admissions'], 's-', color='#e34a33', linewidth=2, markersize=6, alpha=0.7)
ax2.set_ylabel('Admissions', color='#e34a33')
ax2.set_title('Monthly Healthcare Cost & Admission Trend (2024)', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('output/monthly_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ output/monthly_trend.png")

# Fig 3: Risk Score Heatmap
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

risk_summary = pd.DataFrame({
    'risk_range': ['Low', 'Medium', 'High', 'Very High'],
    'avg_cost': [cost_by_risk.loc[r, 'avg_cost'] for r in cost_by_risk.index],
    'admit_rate': [cost_by_risk.loc[r, 'admission_rate'] * 100 for r in cost_by_risk.index],
    'cost_per_patient': [cost_by_risk.loc[r, 'cost_per_patient'] for r in cost_by_risk.index]
})

colors = sns.color_palette('mako_r', 4)
axes[0].bar(risk_summary['risk_range'], risk_summary['avg_cost'], color=colors)
axes[0].set_title('Avg Cost per Claim', fontweight='bold')
axes[0].set_ylabel('$')

axes[1].bar(risk_summary['risk_range'], risk_summary['admit_rate'], color=colors)
axes[1].set_title('Admission Rate (%)', fontweight='bold')
axes[1].set_ylabel('%')

axes[2].bar(risk_summary['risk_range'], risk_summary['cost_per_patient'], color=colors)
axes[2].set_title('Cost per Patient ($)', fontweight='bold')
axes[2].set_ylabel('$')

fig.suptitle('Impact of Patient Risk Score on Healthcare Utilisation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/risk_score_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ output/risk_score_impact.png")

# Fig 4: Regional Comparison
region_stats = claims.groupby('patient_region').agg(
    total_cost=('total_cost', 'sum'),
    patient_count=('patient_id', 'nunique'),
    admit_rate=('admitted', 'mean'),
    avg_los=('length_of_stay', 'mean')
).assign(cost_per_patient=lambda x: (x['total_cost'] / x['patient_count']).round(0))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
regions = region_stats.index
colors_r = sns.color_palette('mako', len(regions))

axes[0,0].bar(regions, region_stats['total_cost'] / 1e6, color=colors_r)
axes[0,0].set_title('Total Cost ($M)', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=30)

axes[0,1].bar(regions, region_stats['cost_per_patient'] / 1000, color=colors_r)
axes[0,1].set_title('Cost per Patient ($K)', fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=30)

axes[1,0].bar(regions, region_stats['admit_rate'] * 100, color=colors_r)
axes[1,0].set_title('Admission Rate (%)', fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=30)

axes[1,1].bar(regions, region_stats['avg_los'], color=colors_r)
axes[1,1].set_title('Avg Length of Stay (Days)', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=30)

fig.suptitle('Regional Healthcare Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/regional_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ output/regional_comparison.png")

# ── 5. Predictive Model ───────────────────────────────────
print("\n" + "=" * 60)
print("PREDICTIVE MODEL: Admission Risk Classifier")
print("=" * 60)

# Feature engineering
df = claims.copy()
features = ['patient_age', 'patient_risk_score', 'total_cost']

# Encode categorical features
le_region = LabelEncoder()
le_dx = LabelEncoder()
df['region_enc'] = le_region.fit_transform(df['patient_region'])
df['dx_enc'] = le_dx.fit_transform(df['diagnosis_desc'])
features.extend(['region_enc', 'dx_enc'])

X = df[features]
y = df['admitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"\nTraining samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Baseline accuracy (predict 0): {1 - y_test.mean():.3f}")
print(f"Model accuracy: {(y_pred == y_test).mean():.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Admitted', 'Admitted']))

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

print("\nFeature Importance:")
for _, row in importance.iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

# Feature importance chart
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(importance['feature'], importance['importance'], color=sns.color_palette('mako', len(features)))
ax.set_title('Feature Importance — Admission Risk Prediction', fontweight='bold')
ax.set_xlabel('Importance')
for bar, val in zip(bars, importance['importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ output/feature_importance.png")

# ── Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"""
Key Findings:
  1. Total claims cost: ${total_cost:,.0f}
  2. {admit_rate:.1f}% of claims result in hospital admission
  3. High-risk patients cost {cost_by_risk.loc['High (2.5-3.5)', 'cost_per_patient']:,.0f}/patient
     vs {cost_by_risk.loc['Low (<1.5)', 'cost_per_patient']:,.0f}/patient for low-risk
  4. Readmission rate: {readmitted_patients/total_admitted_patients*100:.1f}% of admitted patients
  5. Model predicts admission risk with ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}

Files generated:
  - output/diagnosis_cost_analysis.png
  - output/monthly_trend.png
  - output/risk_score_impact.png
  - output/regional_comparison.png
  - output/feature_importance.png
""")
