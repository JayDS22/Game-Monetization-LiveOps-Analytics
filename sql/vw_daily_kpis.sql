
-- Daily KPI aggregation view (production DWH simulation)
CREATE OR REPLACE VIEW vw_daily_kpis AS
SELECT 
    DATE(last_active_date) AS activity_date,
    COUNT(DISTINCT player_id) AS dau,
    COUNT(DISTINCT CASE WHEN is_payer = 1 THEN player_id END) AS paying_dau,
    SUM(total_spend_usd) AS daily_revenue,
    AVG(total_sessions) AS avg_sessions,
    AVG(avg_session_minutes) AS avg_session_length
FROM players
GROUP BY DATE(last_active_date);
