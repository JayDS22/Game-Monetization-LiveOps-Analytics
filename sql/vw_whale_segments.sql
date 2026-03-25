
-- Whale segmentation view
CREATE OR REPLACE VIEW vw_whale_segments AS
SELECT 
    player_tier,
    COUNT(*) AS player_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_players,
    SUM(total_spend_usd) AS total_revenue,
    ROUND(100.0 * SUM(total_spend_usd) / SUM(SUM(total_spend_usd)) OVER(), 2) AS pct_revenue,
    AVG(total_spend_usd) AS avg_spend,
    AVG(total_sessions) AS avg_sessions,
    AVG(days_active) AS avg_days_active,
    AVG(max_stage_reached) AS avg_max_stage
FROM players
WHERE is_payer = 1
GROUP BY player_tier;
