-- Unit-economics views (Objective 2).
-- These build reusable CAC / ROAS / LTV:CAC views on top of the raw tables, the
-- way the metrics would live in a warehouse for BI tools to sit on. Run by
-- sql_views.py, which loads data/*.csv into SQLite, creates the views and
-- materialises each SELECT to sql_output/.

-- name: v_channel_media  [view]
-- Media spend and clicks rolled up per channel.
CREATE VIEW v_channel_media AS
SELECT channel,
       SUM(spend)  AS spend,
       SUM(clicks) AS clicks
FROM spend
GROUP BY channel;

-- name: v_channel_customers  [view]
-- Signups, paying customers and revenue per channel (paying customers only for
-- revenue / lifetime value, so the economics aren't diluted by non-activators).
CREATE VIEW v_channel_customers AS
SELECT channel,
       COUNT(*)                                          AS signups,
       SUM(converted)                                    AS customers,
       SUM(commission_revenue)                           AS revenue,
       SUM(CASE WHEN converted = 1 THEN ltv ELSE 0 END)  AS ltv_sum
FROM users
GROUP BY channel;

-- name: v_channel_unit_economics  [view]
-- The headline per-channel unit-economics view: CAC, ROAS, ARPC and LTV:CAC.
CREATE VIEW v_channel_unit_economics AS
SELECT m.channel,
       ROUND(m.spend)                                  AS spend,
       c.customers,
       ROUND(m.spend * 1.0 / c.customers, 2)           AS cac,
       ROUND(c.revenue * 1.0 / c.customers, 2)         AS arpc,
       ROUND((c.ltv_sum / c.customers)
             / (m.spend * 1.0 / c.customers), 2)       AS ltv_cac,
       ROUND(c.revenue * 1.0 / m.spend, 2)             AS roas
FROM v_channel_media m
JOIN v_channel_customers c USING (channel)
ORDER BY roas DESC;

-- name: channel_unit_economics
-- Materialise the channel view.
SELECT * FROM v_channel_unit_economics;

-- name: country_unit_economics
-- The same economics cut by market instead of channel.
WITH media AS (
    SELECT country, SUM(spend) AS spend FROM spend GROUP BY country
),
cust AS (
    SELECT country,
           SUM(converted)                                   AS customers,
           SUM(commission_revenue)                          AS revenue,
           SUM(CASE WHEN converted = 1 THEN ltv ELSE 0 END) AS ltv_sum
    FROM users GROUP BY country
)
SELECT m.country,
       ROUND(m.spend)                            AS spend,
       c.customers,
       ROUND(m.spend * 1.0 / c.customers, 2)     AS cac,
       ROUND((c.ltv_sum / c.customers)
             / (m.spend * 1.0 / c.customers), 2) AS ltv_cac,
       ROUND(c.revenue * 1.0 / m.spend, 2)       AS roas
FROM media m JOIN cust c USING (country)
ORDER BY roas DESC;

-- name: channel_country_cac  [matrix]
-- CAC for every channel x market pair -- feeds the heatmap and surfaces pockets
-- of waste that a channel-only or market-only view would hide.
WITH media AS (
    SELECT channel, country, SUM(spend) AS spend
    FROM spend GROUP BY channel, country
),
cust AS (
    SELECT channel, country, SUM(converted) AS customers
    FROM users GROUP BY channel, country
)
SELECT m.channel,
       m.country,
       ROUND(m.spend / NULLIF(c.customers, 0), 2) AS cac
FROM media m JOIN cust c USING (channel, country)
ORDER BY m.channel, m.country;
