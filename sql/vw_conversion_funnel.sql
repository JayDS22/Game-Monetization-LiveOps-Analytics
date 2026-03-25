
-- F2P conversion funnel view
CREATE OR REPLACE VIEW vw_conversion_funnel AS
SELECT
    'install' AS stage, COUNT(*) AS players FROM players
UNION ALL
SELECT 'tutorial_complete', COUNT(*) FROM players WHERE tutorial_completed = 1
UNION ALL
SELECT 'd1_retained', COUNT(*) FROM players WHERE d1_retained = 1
UNION ALL
SELECT 'd7_retained', COUNT(*) FROM players WHERE d7_retained = 1
UNION ALL
SELECT 'first_iap', COUNT(*) FROM players WHERE is_payer = 1
UNION ALL
SELECT 'repeat_purchaser', COUNT(*) FROM players WHERE num_purchases >= 2;
