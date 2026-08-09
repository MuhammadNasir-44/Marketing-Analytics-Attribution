-- Retention, LTV and payback queries in SQL (Objective 2, Day 4).
-- Run by sql_cohorts.py, which loads data/*.csv into SQLite. Only paying
-- customers (converted = 1) carry a tenure and an LTV, so the cohort work is
-- scoped to them. Each query is named with a `-- name:` header.

-- name: cohort_sizes
-- Number of paying customers acquired each signup month (the cohorts).
SELECT strftime('%Y-%m', signup_date) AS cohort,
       COUNT(*)                        AS customers
FROM users
WHERE converted = 1
GROUP BY cohort
ORDER BY cohort;

-- name: retention_by_tenure
-- Retention curve: share of customers still active at each tenure month, using
-- an at-risk denominator (only customers whose cohort is old enough to have been
-- observed at that tenure, measured against 2026-06-30). This is the SQL twin of
-- retention_curves_by_channel() in ltv_retention.py.
WITH cust AS (
    SELECT months_active,
           (2026 - CAST(strftime('%Y', signup_date) AS INT)) * 12
           + (6   - CAST(strftime('%m', signup_date) AS INT)) AS cohort_age
    FROM users
    WHERE converted = 1
),
tenures AS (
    SELECT 0 AS t UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3
    UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7
    UNION ALL SELECT 8 UNION ALL SELECT 9 UNION ALL SELECT 10 UNION ALL SELECT 11
)
SELECT t.t AS tenure_month,
       ROUND(AVG(CASE WHEN c.months_active > t.t THEN 100.0 ELSE 0.0 END), 1)
           AS retention_pct,
       COUNT(*) AS at_risk
FROM tenures t
JOIN cust c ON c.cohort_age >= t.t
GROUP BY t.t
ORDER BY t.t;

-- name: ltv_by_channel
-- Average predicted LTV, observed revenue and tenure per acquisition channel.
SELECT channel,
       COUNT(*)                        AS customers,
       ROUND(AVG(ltv), 2)              AS avg_ltv,
       ROUND(AVG(commission_revenue), 2) AS avg_observed_rev,
       ROUND(AVG(months_active), 2)    AS avg_tenure
FROM users
WHERE converted = 1
GROUP BY channel
ORDER BY avg_ltv DESC;

-- name: payback_and_ltv_cac
-- CAC, monthly ARPU, payback period (months) and LTV:CAC per channel, joining
-- media spend to paying-customer revenue.
WITH media AS (
    SELECT channel, SUM(spend) AS spend FROM spend GROUP BY channel
),
cust AS (
    SELECT channel,
           COUNT(*)                  AS customers,
           SUM(commission_revenue)   AS revenue,
           SUM(months_active)        AS active_months,
           AVG(ltv)                  AS avg_ltv
    FROM users
    WHERE converted = 1
    GROUP BY channel
)
SELECT c.channel,
       ROUND(m.spend / c.customers, 2)                             AS cac,
       ROUND(c.revenue / c.active_months, 2)                       AS monthly_arpu,
       ROUND((m.spend / c.customers) / (c.revenue / c.active_months), 2)
                                                                   AS payback_months,
       ROUND(c.avg_ltv / (m.spend / c.customers), 2)               AS ltv_cac
FROM media m JOIN cust c USING (channel)
ORDER BY payback_months;
