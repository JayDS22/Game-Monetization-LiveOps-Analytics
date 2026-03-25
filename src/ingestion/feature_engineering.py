"""
Module 1: Data Ingestion & Feature Engineering Pipeline
========================================================
- Load and join datasets
- Engineer 40+ behavioral/spending/engagement features
- SQL-style views for production data warehouse simulation
"""

import pandas as pd
import numpy as np
import os

RAW = "/home/claude/game-monetization/data/raw"
PROCESSED = "/home/claude/game-monetization/data/processed"
os.makedirs(PROCESSED, exist_ok=True)

print("=" * 60)
print("MODULE 1: FEATURE ENGINEERING PIPELINE")
print("=" * 60)

# ── Load raw data ──
players = pd.read_csv(f"{RAW}/players.csv", parse_dates=['install_date', 'last_active_date'])
txns = pd.read_csv(f"{RAW}/transactions.csv", parse_dates=['transaction_date'])
ab = pd.read_csv(f"{RAW}/cookie_cats_ab.csv")

print(f"Players: {len(players):,}")
print(f"Transactions: {len(txns):,}")
print(f"A/B test: {len(ab):,}")

# ── Transaction-level features ──
print("\nEngineering transaction features...")

txn_features = txns.groupby('player_id').agg(
    first_purchase_date=('transaction_date', 'min'),
    last_purchase_date=('transaction_date', 'max'),
    total_txn_count=('transaction_id', 'count'),
    total_revenue=('price_usd', 'sum'),
    avg_txn_value=('price_usd', 'mean'),
    max_txn_value=('price_usd', 'max'),
    min_txn_value=('price_usd', 'min'),
    std_txn_value=('price_usd', 'std'),
    unique_items_purchased=('item_name', 'nunique'),
    unique_categories=('item_category', 'nunique'),
).reset_index()

# Purchase velocity (txns per active day)
txn_date_range = txns.groupby('player_id').agg(
    purchase_span_days=('transaction_date', lambda x: (x.max() - x.min()).days + 1)
).reset_index()
txn_features = txn_features.merge(txn_date_range, on='player_id', how='left')
txn_features['purchase_velocity'] = txn_features['total_txn_count'] / txn_features['purchase_span_days'].clip(lower=1)

# Category spend breakdown
cat_spend = txns.pivot_table(index='player_id', columns='item_category', values='price_usd', aggfunc='sum', fill_value=0)
cat_spend.columns = [f'spend_{c}' for c in cat_spend.columns]
cat_spend = cat_spend.reset_index()
txn_features = txn_features.merge(cat_spend, on='player_id', how='left')

# ── RFM features ──
print("Computing RFM scores...")
observation_date = pd.Timestamp('2025-01-15')

rfm = txns.groupby('player_id').agg(
    recency=('transaction_date', lambda x: (observation_date - x.max()).days),
    frequency=('transaction_id', 'count'),
    monetary=('price_usd', 'sum'),
).reset_index()

# RFM quintile scoring
for col in ['recency', 'frequency', 'monetary']:
    if col == 'recency':
        rfm[f'{col}_score'] = pd.qcut(rfm[col], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    else:
        rfm[f'{col}_score'] = pd.qcut(rfm[col].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)

rfm['rfm_score'] = rfm['recency_score'] * 100 + rfm['frequency_score'] * 10 + rfm['monetary_score']

txn_features = txn_features.merge(rfm[['player_id', 'recency', 'frequency', 'monetary', 
                                        'recency_score', 'frequency_score', 'monetary_score', 'rfm_score']], 
                                   on='player_id', how='left')

# ── Merge all features into master table ──
print("Building master feature table...")

df = players.copy()

# Merge transaction features (left join - non-payers will have NaN)
df = df.merge(txn_features, on='player_id', how='left')

# ── Derived engagement features ──
df['days_since_install'] = (observation_date - df['install_date']).dt.days
df['days_since_last_active'] = (observation_date - df['last_active_date']).dt.days
df['session_frequency'] = df['total_sessions'] / df['days_since_install'].clip(lower=1)
df['play_intensity'] = df['total_play_hours'] / df['days_active'].clip(lower=1)
df['stage_velocity'] = df['max_stage_reached'] / df['days_active'].clip(lower=1)
df['engagement_ratio'] = df['days_active'] / df['days_since_install'].clip(lower=1)

# Play streak estimation (simulated)
np.random.seed(42)
df['max_play_streak'] = np.clip(
    (df['engagement_ratio'] * 15 + np.random.exponential(2, len(df))).astype(int), 0, 60
)
df['avg_play_streak'] = (df['max_play_streak'] * 0.4 + np.random.exponential(1, len(df))).round(1)

# Time-to-first-purchase (in hours, for payers)
df['hours_to_first_purchase'] = df['days_to_first_purchase'] * 24

# Social features
df['social_multiplier'] = 1 + 0.2 * df['fb_connected']

# Platform encoding
df['is_ios'] = (df['platform'] == 'iOS').astype(int)

# Country tier (for spending power)
high_spend_countries = ['US', 'JP', 'KR', 'UK', 'DE', 'AU', 'CA', 'SG', 'TW', 'FR']
df['is_high_spend_country'] = df['country'].isin(high_spend_countries).astype(int)

# Age buckets
df['age_bucket'] = pd.cut(df['age'], bins=[0, 17, 24, 34, 44, 100], labels=['teen', 'young_adult', 'adult', 'middle_aged', 'senior'])

# Progression efficiency
df['stage_per_session'] = df['max_stage_reached'] / df['total_sessions'].clip(lower=1)
df['stage_per_hour'] = df['max_stage_reached'] / df['total_play_hours'].clip(lower=0.1)

# Revenue per session / per hour (for payers)
df['revenue_per_session'] = df['total_spend_usd'] / df['total_sessions'].clip(lower=1)
df['revenue_per_hour'] = df['total_spend_usd'] / df['total_play_hours'].clip(lower=0.1)
df['revenue_per_active_day'] = df['total_spend_usd'] / df['days_active'].clip(lower=1)

# Retention trajectory score (weighted sum)
df['retention_score'] = df['d1_retained'] * 1 + df['d7_retained'] * 3 + df['d30_retained'] * 7

# Fill NaN for non-payers
fill_zero_cols = [c for c in df.columns if 'spend_' in c or 'revenue' in c.lower() or 'txn' in c.lower() 
                  or 'rfm' in c.lower() or c.startswith('recency') or c.startswith('frequency') 
                  or c.startswith('monetary') or 'purchase' in c.lower()]
for c in fill_zero_cols:
    if c in df.columns:
        df[c] = df[c].fillna(0)

# ── Save processed dataset ──
df.to_csv(f"{PROCESSED}/master_features.csv", index=False)
print(f"\nMaster feature table saved: {len(df):,} rows, {len(df.columns)} columns")

# ── Feature summary ──
feature_cols = [c for c in df.columns if c not in ['player_id', 'install_date', 'last_active_date',
                                                      'platform', 'country', 'gender', 'player_tier', 'age_bucket']]
print(f"\nTotal engineered features: {len(feature_cols)}")
print("\nFeature categories:")
print(f"  Engagement: session_frequency, play_intensity, stage_velocity, engagement_ratio, max_play_streak, ...")
print(f"  Spending: total_spend_usd, avg_txn_value, purchase_velocity, revenue_per_session, ...")
print(f"  RFM: recency, frequency, monetary, rfm_score, recency_score, ...")
print(f"  Retention: d1_retained, d7_retained, d30_retained, retention_score, ...")
print(f"  Progression: max_stage_reached, stage_per_session, stage_per_hour, ...")
print(f"  Demographics: age, is_ios, is_high_spend_country, fb_connected, ...")

# ── SQL Views (written as .sql files) ──
print("\nGenerating SQL view definitions...")

sql_dir = "/home/claude/game-monetization/sql"
os.makedirs(sql_dir, exist_ok=True)

sql_views = {
    "vw_daily_kpis.sql": """
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
""",
    "vw_cohort_retention.sql": """
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
""",
    "vw_whale_segments.sql": """
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
""",
    "vw_conversion_funnel.sql": """
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
""",
}

for fname, sql in sql_views.items():
    with open(f"{sql_dir}/{fname}", 'w') as f:
        f.write(sql)
    print(f"  Saved: {fname}")

print("\n✓ Module 1 complete!")
