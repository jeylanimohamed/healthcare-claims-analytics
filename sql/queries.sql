-- Healthcare Claims Analytics — SQL Queries
-- Generated from 25,000 claims across 5,000 patients and 200 providers
-- Run: sqlite3 data/healthcare.db < sql/queries.sql  (after loading CSVs)

-- ============================
-- 1. Total Cost by Diagnosis
-- ============================
SELECT
    diagnosis_desc,
    COUNT(*) AS claim_count,
    ROUND(AVG(total_cost), 2) AS avg_cost,
    ROUND(SUM(total_cost), 2) AS total_cost,
    ROUND(SUM(total_cost) * 100.0 / (SELECT SUM(total_cost) FROM claims), 1) AS pct_of_total
FROM claims
GROUP BY diagnosis_desc
ORDER BY total_cost DESC;

-- ============================
-- 2. Monthly Cost Trend (2024)
-- ============================
SELECT
    strftime('%Y-%m', service_date) AS month,
    COUNT(*) AS claim_count,
    ROUND(SUM(total_cost), 2) AS total_cost,
    ROUND(AVG(total_cost), 2) AS avg_cost_per_claim,
    COUNT(DISTINCT patient_id) AS unique_patients
FROM claims
GROUP BY month
ORDER BY month;

-- ============================
-- 3. Top 10 Highest-Cost Patients
-- ============================
SELECT
    c.patient_id,
    p.age,
    p.gender,
    p.risk_score,
    p.num_chronic_conditions,
    p.chronic_conditions,
    COUNT(c.claim_id) AS claim_count,
    ROUND(SUM(c.total_cost), 2) AS total_cost,
    ROUND(AVG(c.total_cost), 2) AS avg_cost_per_claim,
    SUM(c.admitted) AS admissions,
    SUM(c.length_of_stay) AS total_los_days
FROM claims c
JOIN patients p ON c.patient_id = p.patient_id
GROUP BY c.patient_id
ORDER BY total_cost DESC
LIMIT 10;

-- ============================
-- 4. Provider Specialty Performance
-- ============================
SELECT
    c.provider_specialty,
    COUNT(DISTINCT c.provider_id) AS provider_count,
    COUNT(c.claim_id) AS total_claims,
    ROUND(SUM(c.total_cost), 2) AS total_cost,
    ROUND(AVG(c.total_cost), 2) AS avg_cost_per_claim,
    ROUND(SUM(c.admitted) * 100.0 / COUNT(c.claim_id), 1) AS admission_rate_pct,
    ROUND(AVG(c.length_of_stay), 1) AS avg_los
FROM claims c
GROUP BY c.provider_specialty
ORDER BY total_cost DESC;

-- ============================
-- 5. High-Risk Patients (risk_score > 2.5) — Cost Analysis
-- ============================
SELECT
    CASE
        WHEN p.risk_score > 3.5 THEN 'Very High (>3.5)'
        WHEN p.risk_score > 2.5 THEN 'High (2.5-3.5)'
        WHEN p.risk_score > 1.5 THEN 'Medium (1.5-2.5)'
        ELSE 'Low (<1.5)'
    END AS risk_category,
    COUNT(DISTINCT c.patient_id) AS patient_count,
    COUNT(c.claim_id) AS claim_count,
    ROUND(SUM(c.total_cost), 2) AS total_cost,
    ROUND(AVG(c.total_cost), 2) AS avg_cost_per_claim,
    ROUND(SUM(c.total_cost) / COUNT(DISTINCT c.patient_id), 2) AS cost_per_patient
FROM claims c
JOIN patients p ON c.patient_id = p.patient_id
GROUP BY risk_category
ORDER BY AVG(p.risk_score) DESC;

-- ============================
-- 6. Readmission Analysis (patients with 2+ admissions in 2024)
-- ============================
SELECT
    c.patient_id,
    p.age,
    p.risk_score,
    p.num_chronic_conditions,
    SUM(c.admitted) AS total_admissions,
    SUM(c.length_of_stay) AS total_los_days,
    ROUND(SUM(c.total_cost), 2) AS total_cost
FROM claims c
JOIN patients p ON c.patient_id = p.patient_id
WHERE c.admitted = 1
GROUP BY c.patient_id
HAVING total_admissions >= 2
ORDER BY total_admissions DESC, total_cost DESC
LIMIT 20;

-- ============================
-- 7. Regional Cost Comparison
-- ============================
SELECT
    c.patient_region,
    COUNT(DISTINCT c.patient_id) AS patient_count,
    COUNT(c.claim_id) AS claim_count,
    ROUND(SUM(c.total_cost), 2) AS total_cost,
    ROUND(AVG(c.total_cost), 2) AS avg_cost_per_claim,
    ROUND(SUM(c.total_cost) / COUNT(DISTINCT c.patient_id), 2) AS cost_per_patient,
    ROUND(SUM(c.admitted) * 100.0 / COUNT(c.claim_id), 1) AS admission_rate_pct
FROM claims c
GROUP BY c.patient_region
ORDER BY total_cost DESC;
