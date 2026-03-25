"""
Synthetic Data Generator for Game Monetization & Live-Ops Analytics Platform
=============================================================================
Generates 3 realistic datasets:
1. Uken-style F2P game data (300K+ players) - engagement, revenue, demographics, progression
2. Mobile IAP transaction data - detailed purchase events
3. Cookie Cats-style A/B test data - gate placement experiment with retention
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

OUTPUT_DIR = "/home/claude/game-monetization/data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_PLAYERS = 310_000
INSTALL_START = datetime(2024, 1, 1)
INSTALL_END = datetime(2024, 12, 31)

print("=" * 60)
print("GENERATING SYNTHETIC F2P GAME DATASETS")
print("=" * 60)

# ── 1. Player Profiles (Uken-style) ──────────────────────────────────────
print("\n[1/3] Generating player profiles (310K players)...")

install_dates = pd.to_datetime(
    np.random.uniform(INSTALL_START.timestamp(), INSTALL_END.timestamp(), N_PLAYERS),
    unit='s'
).normalize()

platforms = np.random.choice(['iOS', 'Android'], N_PLAYERS, p=[0.45, 0.55])
countries = np.random.choice(
    ['US', 'UK', 'CA', 'DE', 'JP', 'KR', 'BR', 'IN', 'FR', 'AU', 'CN', 'TW', 'SG', 'MX', 'ID'],
    N_PLAYERS,
    p=[0.25, 0.08, 0.06, 0.07, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.06, 0.03, 0.02, 0.04, 0.03]
)
genders = np.random.choice(['M', 'F', 'Other'], N_PLAYERS, p=[0.52, 0.44, 0.04])
ages = np.clip(np.random.lognormal(3.3, 0.35, N_PLAYERS).astype(int), 13, 65)
fb_connected = np.random.binomial(1, 0.32, N_PLAYERS)

# Spending propensity - power law (whale/dolphin/minnow distribution)
spending_propensity = np.random.pareto(2.5, N_PLAYERS)
spending_propensity = spending_propensity / spending_propensity.max()

# Higher propensity for certain countries/platforms
country_boost = pd.Series(countries).map({
    'US': 1.3, 'JP': 1.5, 'KR': 1.4, 'UK': 1.1, 'DE': 1.1,
    'CN': 1.2, 'TW': 1.2, 'CA': 1.1, 'AU': 1.1, 'SG': 1.2,
    'BR': 0.7, 'IN': 0.5, 'FR': 0.9, 'MX': 0.6, 'ID': 0.4
}).values
spending_propensity *= country_boost

# FB connected players spend ~20% more
spending_propensity *= (1 + 0.2 * fb_connected)

# Engagement level correlates with spending
engagement_base = np.clip(np.random.beta(2, 5, N_PLAYERS) + 0.3 * spending_propensity, 0, 1)

# Session & engagement metrics
total_sessions = np.clip(
    (engagement_base * 200 + np.random.exponential(10, N_PLAYERS)).astype(int), 1, 500
)
avg_session_minutes = np.clip(
    engagement_base * 25 + np.random.normal(5, 3, N_PLAYERS), 1, 90
).round(1)
total_play_hours = (total_sessions * avg_session_minutes / 60).round(1)

# Days active & retention
max_days = (INSTALL_END - install_dates).days.values.astype(float)
max_days = np.clip(max_days, 1, 365)
days_active = np.clip(
    (engagement_base * max_days * 0.6 + np.random.exponential(5, N_PLAYERS)).astype(int), 1, max_days
).astype(int)

# D1, D7, D30 retention (realistic F2P retention curves)
d1_retained = np.random.binomial(1, np.clip(0.40 + 0.3 * engagement_base, 0, 0.85), N_PLAYERS)
d7_retained = np.random.binomial(1, np.clip(0.15 + 0.35 * engagement_base, 0, 0.65), N_PLAYERS) * d1_retained
d7_retained = np.where(d1_retained == 0, np.random.binomial(1, 0.05, N_PLAYERS), d7_retained)
d30_retained = np.random.binomial(1, np.clip(0.06 + 0.25 * engagement_base, 0, 0.45), N_PLAYERS) * d7_retained
d30_retained = np.where(d7_retained == 0, np.random.binomial(1, 0.02, N_PLAYERS), d30_retained)

# Stage / level progression
max_stage = np.clip(
    (engagement_base * 150 + spending_propensity * 30 + np.random.exponential(5, N_PLAYERS)).astype(int),
    1, 200
)

# Tutorial completion
tutorial_completed = np.random.binomial(1, np.clip(0.70 + 0.2 * engagement_base, 0, 0.98), N_PLAYERS)

# IAP conversion (realistic ~3-5% conversion rate, whales are tiny subset)
iap_probability = np.clip(0.02 + 0.12 * spending_propensity + 0.04 * engagement_base, 0, 0.95)
is_payer = np.random.binomial(1, iap_probability, N_PLAYERS)
payer_rate = is_payer.mean()
print(f"  Payer conversion rate: {payer_rate:.1%}")

# Total spend for payers (power law)
total_spend = np.zeros(N_PLAYERS)
payer_mask = is_payer == 1
n_payers = payer_mask.sum()
total_spend[payer_mask] = np.random.pareto(1.2, n_payers) * 5 + 0.99
total_spend[payer_mask] = np.clip(total_spend[payer_mask], 0.99, 15000)
total_spend = total_spend.round(2)

# Number of purchases for payers
n_purchases = np.zeros(N_PLAYERS, dtype=int)
n_purchases[payer_mask] = np.clip(
    (np.log1p(total_spend[payer_mask]) * 1.5 + np.random.exponential(1, n_payers)).astype(int), 1, 200
)

# Time to first purchase (days from install)
days_to_first_purchase = np.full(N_PLAYERS, np.nan)
days_to_first_purchase[payer_mask] = np.clip(
    np.random.exponential(7, n_payers), 0, max_days[payer_mask]
).astype(int)

# Last active date - some players are still active recently
observation_date = datetime(2025, 1, 15)
recency_days = np.zeros(N_PLAYERS)
# ~35% of players are recently active (within 7 days of observation)
still_active_mask = np.random.binomial(1, np.clip(0.15 + 0.4 * engagement_base, 0, 0.85), N_PLAYERS).astype(bool)
recency_days[still_active_mask] = np.random.randint(0, 6, still_active_mask.sum())
recency_days[~still_active_mask] = np.clip(
    np.random.exponential(30, (~still_active_mask).sum()) + 7, 7, 365
).astype(int)
last_active = pd.to_datetime(observation_date) - pd.to_timedelta(recency_days, unit='D')
# Don't let last_active be before install
last_active = np.maximum(last_active, install_dates + pd.to_timedelta(1, unit='D'))

# Churn label (inactive for 7+ days from "today" = 2025-01-15)
days_since_last_active = (pd.to_datetime(observation_date) - last_active).days
churned_d7 = (days_since_last_active >= 7).astype(int)
print(f"  D7 churn rate: {churned_d7.mean():.1%}")

players_df = pd.DataFrame({
    'player_id': [f'P{i:06d}' for i in range(N_PLAYERS)],
    'install_date': install_dates,
    'last_active_date': last_active,
    'platform': platforms,
    'country': countries,
    'gender': genders,
    'age': ages,
    'fb_connected': fb_connected,
    'tutorial_completed': tutorial_completed,
    'total_sessions': total_sessions,
    'avg_session_minutes': avg_session_minutes,
    'total_play_hours': total_play_hours,
    'days_active': days_active,
    'max_stage_reached': max_stage,
    'd1_retained': d1_retained,
    'd7_retained': d7_retained,
    'd30_retained': d30_retained,
    'is_payer': is_payer,
    'total_spend_usd': total_spend,
    'num_purchases': n_purchases,
    'days_to_first_purchase': days_to_first_purchase,
    'churned_d7': churned_d7,
})

# Classify player tiers
def classify_tier(row):
    if row['is_payer'] == 0:
        return 'free_rider'
    spend_pctile = total_spend[total_spend > 0]
    if row['total_spend_usd'] >= np.percentile(spend_pctile, 97):
        return 'whale'
    elif row['total_spend_usd'] >= np.percentile(spend_pctile, 80):
        return 'dolphin'
    else:
        return 'minnow'

players_df['player_tier'] = players_df.apply(classify_tier, axis=1)
print(f"  Tier distribution:\n{players_df['player_tier'].value_counts()}")

players_df.to_csv(f"{OUTPUT_DIR}/players.csv", index=False)
print(f"  Saved: players.csv ({len(players_df):,} rows)")

# ── 2. IAP Transaction Data ────────────────────────────────────────────────
print("\n[2/3] Generating IAP transaction data...")

iap_items = [
    ('gem_pack_small', 0.99, 'currency'), ('gem_pack_medium', 4.99, 'currency'),
    ('gem_pack_large', 9.99, 'currency'), ('gem_pack_mega', 49.99, 'currency'),
    ('gem_pack_ultra', 99.99, 'currency'),
    ('starter_bundle', 2.99, 'bundle'), ('weekly_bundle', 4.99, 'bundle'),
    ('monthly_bundle', 9.99, 'bundle'), ('vip_pass', 14.99, 'subscription'),
    ('battle_pass', 9.99, 'subscription'), ('season_pass', 19.99, 'subscription'),
    ('energy_refill', 0.99, 'consumable'), ('extra_lives', 1.99, 'consumable'),
    ('speed_boost', 2.99, 'consumable'), ('shield_pack', 3.99, 'consumable'),
    ('cosmetic_skin_1', 4.99, 'cosmetic'), ('cosmetic_skin_2', 9.99, 'cosmetic'),
    ('legendary_chest', 14.99, 'lootbox'), ('epic_chest', 7.99, 'lootbox'),
    ('rare_chest', 2.99, 'lootbox'),
]

transactions = []
payer_ids = players_df[players_df['is_payer'] == 1]['player_id'].values

for _, row in players_df[players_df['is_payer'] == 1].iterrows():
    pid = row['player_id']
    n_txns = row['num_purchases']
    install = row['install_date']
    first_purchase_day = row['days_to_first_purchase']
    tier = row['player_tier']

    # Item selection weights based on tier
    if tier == 'whale':
        weights = [0.02, 0.05, 0.10, 0.18, 0.15, 0.03, 0.05, 0.08, 0.08, 0.06, 0.05, 0.01, 0.01, 0.02, 0.01, 0.02, 0.03, 0.03, 0.01, 0.01]
    elif tier == 'dolphin':
        weights = [0.05, 0.12, 0.15, 0.08, 0.02, 0.06, 0.08, 0.06, 0.06, 0.08, 0.04, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02, 0.02, 0.02, 0.01]
    else:
        weights = [0.20, 0.15, 0.08, 0.01, 0.00, 0.08, 0.06, 0.03, 0.03, 0.04, 0.02, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01, 0.03, 0.02]
    
    weights = np.array(weights)
    weights = weights / weights.sum()

    for txn_i in range(n_txns):
        item_idx = np.random.choice(len(iap_items), p=weights)
        item_name, base_price, category = iap_items[item_idx]
        
        # Some price variation (sales, bundles)
        price_mult = np.random.choice([1.0, 0.8, 0.5], p=[0.80, 0.15, 0.05])
        price = round(base_price * price_mult, 2)
        
        # Transaction timing
        if txn_i == 0:
            day_offset = int(first_purchase_day) if not np.isnan(first_purchase_day) else 1
        else:
            day_offset = int(first_purchase_day + np.random.exponential(14) * (txn_i + 1) / n_txns) if not np.isnan(first_purchase_day) else txn_i + 1
        
        txn_date = install + timedelta(days=min(day_offset, 365))
        
        transactions.append({
            'transaction_id': f'TXN{len(transactions):08d}',
            'player_id': pid,
            'transaction_date': txn_date,
            'item_name': item_name,
            'item_category': category,
            'price_usd': price,
            'platform': row['platform'],
            'country': row['country'],
        })

transactions_df = pd.DataFrame(transactions)
transactions_df = transactions_df.sort_values('transaction_date').reset_index(drop=True)
print(f"  Total transactions: {len(transactions_df):,}")
print(f"  Revenue: ${transactions_df['price_usd'].sum():,.2f}")
transactions_df.to_csv(f"{OUTPUT_DIR}/transactions.csv", index=False)
print(f"  Saved: transactions.csv")

# ── 3. Cookie Cats A/B Test Data ──────────────────────────────────────────
print("\n[3/3] Generating Cookie Cats A/B test data...")

N_AB = 90_000
ab_versions = np.random.choice(['gate_30', 'gate_40'], N_AB, p=[0.5, 0.5])

# Gate 30 has slightly better D1 but gate 40 has slightly better D7
ab_game_rounds = np.random.negative_binomial(5, 0.02, N_AB)
ab_game_rounds = np.clip(ab_game_rounds, 0, 50000)

d1_probs = np.where(ab_versions == 'gate_30', 0.448, 0.442)
d7_probs = np.where(ab_versions == 'gate_30', 0.190, 0.182)

# Higher engagement -> higher retention
engagement_factor = np.clip(ab_game_rounds / 500, 0, 1) * 0.15
d1_probs += engagement_factor
d7_probs += engagement_factor * 0.4

ab_retention_1 = np.random.binomial(1, np.clip(d1_probs, 0, 1), N_AB)
ab_retention_7 = np.random.binomial(1, np.clip(d7_probs, 0, 1), N_AB)

ab_df = pd.DataFrame({
    'userid': range(1, N_AB + 1),
    'version': ab_versions,
    'sum_gamerounds': ab_game_rounds,
    'retention_1': ab_retention_1,
    'retention_7': ab_retention_7,
})

ab_df.to_csv(f"{OUTPUT_DIR}/cookie_cats_ab.csv", index=False)
print(f"  Saved: cookie_cats_ab.csv ({len(ab_df):,} rows)")
print(f"  Gate 30 D1: {ab_df[ab_df['version']=='gate_30']['retention_1'].mean():.3f}")
print(f"  Gate 40 D1: {ab_df[ab_df['version']=='gate_40']['retention_1'].mean():.3f}")
print(f"  Gate 30 D7: {ab_df[ab_df['version']=='gate_30']['retention_7'].mean():.3f}")
print(f"  Gate 40 D7: {ab_df[ab_df['version']=='gate_40']['retention_7'].mean():.3f}")

print("\n" + "=" * 60)
print("ALL DATASETS GENERATED SUCCESSFULLY")
print("=" * 60)
