
-- Cohort retention view
CREATE OR REPLACE VIEW vw_cohort_retention AS
SELECT 
    DATE_TRUNC('week', install_date) AS cohort_week,
    COUNT(DISTINCT player_id) AS cohort_size,
    AVG(d1_retained) AS d1_retention,
    AVG(d7_retained) AS d7_retention,
    AVG(d30_retained) AS d30_retention,
    AVG(CASE WHEN is_payer = 1 THEN total_spend_usd END) AS arppu,
    SUM(total_spend_usd) / COUNT(DISTINCT player_id) AS arpu
FROM players
GROUP BY DATE_TRUNC('week', install_date);
