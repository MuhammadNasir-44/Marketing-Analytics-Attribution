-- Channel-performance analysis queries for the marketing dataset.
-- Three tables are loaded from data/*.csv by sql_analysis.py:
--   spend(date, channel, country, campaign, spend, impressions, clicks)
--   users(user_id, signup_date, country, channel, converted, months_active,
--         commission_revenue, ltv)
--   touchpoints(user_id, seq, channel, touch_date)
-- Each query is named with a `-- name:` header so sql_analysis.py can load it.

-- name: channel_performance
-- Spend, reach and click efficiency per channel (top-of-funnel view).
SELECT channel,
       ROUND(SUM(spend))                                   AS spend,
       SUM(impressions)                                    AS impressions,
       SUM(clicks)                                         AS clicks,
       ROUND(SUM(clicks) * 100.0 / SUM(impressions), 2)    AS ctr_pct,
       ROUND(SUM(spend) / SUM(clicks), 2)                  AS cpc
FROM spend
GROUP BY channel
ORDER BY spend DESC;

-- name: channel_funnel
-- The full acquisition funnel per channel: spend and clicks (from `spend`)
-- joined to signups and paying customers (from `users`), giving click->signup
-- conversion, cost per acquisition and revenue. This is the core Objective 1
-- table -- commercial performance, not vanity metrics.
WITH media AS (
    SELECT channel, SUM(spend) AS spend, SUM(clicks) AS clicks
    FROM spend GROUP BY channel
),
acq AS (
    SELECT channel,
           COUNT(*)                        AS signups,
           SUM(converted)                  AS customers,
           SUM(commission_revenue)         AS revenue
    FROM users GROUP BY channel
)
SELECT m.channel,
       ROUND(m.spend)                                  AS spend,
       a.signups,
       a.customers,
       ROUND(a.signups * 100.0 / m.clicks, 2)          AS click_to_signup_pct,
       ROUND(m.spend / a.customers, 2)                 AS cac,
       ROUND(a.revenue)                                AS revenue,
       ROUND(a.revenue / m.spend, 2)                   AS roas
FROM media m
JOIN acq a USING (channel)
ORDER BY roas DESC;

-- name: country_performance
-- Spend and paying customers by market (multi-market view).
WITH media AS (
    SELECT country, SUM(spend) AS spend FROM spend GROUP BY country
),
acq AS (
    SELECT country, SUM(converted) AS customers, SUM(commission_revenue) AS revenue
    FROM users GROUP BY country
)
SELECT m.country,
       ROUND(m.spend)                    AS spend,
       a.customers,
       ROUND(m.spend / a.customers, 2)   AS cac,
       ROUND(a.revenue / m.spend, 2)     AS roas
FROM media m JOIN acq a USING (country)
ORDER BY roas DESC;

-- name: monthly_spend
-- Monthly spend by channel, to see how the media mix scales over the year.
SELECT strftime('%Y-%m', date) AS month,
       channel,
       ROUND(SUM(spend))        AS spend
FROM spend
GROUP BY month, channel
ORDER BY month, channel;

-- name: top_campaigns
-- The ten highest-spend campaigns, with their click efficiency.
SELECT channel || ' / ' || campaign             AS campaign,
       ROUND(SUM(spend))                        AS spend,
       SUM(clicks)                              AS clicks,
       ROUND(SUM(spend) / SUM(clicks), 2)       AS cpc
FROM spend
GROUP BY channel, campaign
ORDER BY spend DESC
LIMIT 10;
