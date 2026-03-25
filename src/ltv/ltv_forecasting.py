"""
Module 4: LTV Forecasting Models
==================================
- Cox Proportional Hazards for lifetime modeling
- XGBoost for 7/30/90-day LTV prediction
- BG/NBD + Gamma-Gamma for CLV (classic probabilistic approach)
- Model comparison: RMSE, MAE, calibration
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import json, os, warnings
warnings.filterwarnings('ignore')

PROCESSED = "/home/claude/game-monetization/data/processed"
RAW = "/home/claude/game-monetization/data/raw"

print("=" * 60)
print("MODULE 4: LTV FORECASTING MODELS")
print("=" * 60)

df = pd.read_csv(f"{PROCESSED}/segmented_players.csv")
txns = pd.read_csv(f"{RAW}/transactions.csv", parse_dates=['transaction_date'])
print(f"Loaded {len(df):,} players, {len(txns):,} transactions")

# ══════════════════════════════════════════════════════════════
# 1. COX PROPORTIONAL HAZARDS - Player Lifetime Modeling
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("1. COX PROPORTIONAL HAZARDS MODEL")
print("=" * 50)

# Duration = days active, Event = churned
cox_df = df[['days_active', 'churned_d7', 'total_sessions', 'avg_session_minutes',
             'max_stage_reached', 'is_payer', 'tutorial_completed', 'fb_connected',
             'is_ios', 'is_high_spend_country', 'age', 'total_spend_usd']].copy()
cox_df = cox_df.dropna()
cox_df['days_active'] = cox_df['days_active'].clip(lower=1)

# Subsample for memory
np.random.seed(42)
cox_sample = cox_df.sample(n=min(50000, len(cox_df)), random_state=42)

cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_sample, duration_col='days_active', event_col='churned_d7')

print("\nCox PH Model Summary (top features):")
summary = cph.summary
print(summary[['coef', 'exp(coef)', 'p']].sort_values('p').head(10).to_string())
print(f"\nConcordance Index: {cph.concordance_index_:.4f}")

# Kaplan-Meier survival curves by tier
print("\n--- Kaplan-Meier Survival by Player Tier ---")
km_data = {}
kmf = KaplanMeierFitter()
for tier in ['whale', 'dolphin', 'minnow', 'free_rider']:
    tier_df = df[df['player_tier'] == tier]
    tier_sample = tier_df.sample(n=min(5000, len(tier_df)), random_state=42)
    kmf.fit(tier_sample['days_active'].clip(lower=1), tier_sample['churned_d7'], label=tier)
    median_surv = kmf.median_survival_time_
    print(f"  {tier}: median survival = {median_surv:.0f} days")
    km_data[tier] = {
        'timeline': kmf.survival_function_.index.tolist()[:50],
        'survival': kmf.survival_function_[tier].values.tolist()[:50],
        'median': float(median_surv) if not np.isinf(median_surv) else 365.0
    }

# ══════════════════════════════════════════════════════════════
# 2. XGBOOST LTV PREDICTION (7-day, 30-day, 90-day)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("2. XGBOOST LTV PREDICTION")
print("=" * 50)

# Only payers for LTV modeling
payers = df[df['is_payer'] == 1].copy()
print(f"Payers for LTV: {len(payers):,}")

# Simulate multi-horizon LTV targets from transaction data
obs_date = pd.Timestamp('2025-01-15')
install_map = df.set_index('player_id')['install_date'].to_dict()
# Parse install dates
install_map = {k: pd.Timestamp(v) for k, v in install_map.items()}

for horizon, days in [('ltv_7d', 7), ('ltv_30d', 30), ('ltv_90d', 90)]:
    player_ltv = txns.copy()
    player_ltv['install_dt'] = player_ltv['player_id'].map(install_map)
    player_ltv['days_from_install'] = (player_ltv['transaction_date'] - player_ltv['install_dt']).dt.days
    player_ltv = player_ltv[player_ltv['days_from_install'] <= days]
    ltv_vals = player_ltv.groupby('player_id')['price_usd'].sum().reset_index()
    ltv_vals.columns = ['player_id', horizon]
    payers = payers.merge(ltv_vals, on='player_id', how='left')
    payers[horizon] = payers[horizon].fillna(0)

# Features for LTV prediction
ltv_features = [
    'total_sessions', 'avg_session_minutes', 'days_active', 'max_stage_reached',
    'session_frequency', 'play_intensity', 'engagement_ratio',
    'tutorial_completed', 'fb_connected', 'is_ios', 'is_high_spend_country',
    'age', 'd1_retained', 'd7_retained', 'd30_retained',
    'num_purchases', 'total_spend_usd', 'avg_txn_value', 'purchase_velocity',
]

# Ensure all features exist
ltv_features = [f for f in ltv_features if f in payers.columns]

xgb_results = {}
for target in ['ltv_7d', 'ltv_30d', 'ltv_90d']:
    X = payers[ltv_features].fillna(0)
    y = payers[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method='hist'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n  {target.upper()}: RMSE=${rmse:.2f}, MAE=${mae:.2f}, R²={r2:.4f}")
    
    # Feature importance
    imp = pd.Series(model.feature_importances_, index=ltv_features).sort_values(ascending=False)
    print(f"  Top features: {', '.join(imp.head(5).index)}")
    
    xgb_results[target] = {
        'rmse': round(float(rmse), 2),
        'mae': round(float(mae), 2),
        'r2': round(float(r2), 4),
        'feature_importance': {k: round(float(v), 4) for k, v in imp.head(10).items()},
    }

# ══════════════════════════════════════════════════════════════
# 3. BG/NBD + GAMMA-GAMMA (Classic CLV)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("3. BG/NBD + GAMMA-GAMMA CLV MODEL")
print("=" * 50)

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    
    # Prepare RFM summary from transactions
    txns_clean = txns[['player_id', 'transaction_date', 'price_usd']].copy()
    txns_clean.columns = ['customer_id', 'date', 'monetary_value']
    
    rfm_summary = summary_data_from_transaction_data(
        txns_clean, 'customer_id', 'date', 
        monetary_value_col='monetary_value',
        observation_period_end='2025-01-15'
    )
    rfm_summary = rfm_summary[rfm_summary['frequency'] > 0]
    print(f"  RFM summary: {len(rfm_summary):,} customers with repeat purchases")
    
    # BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm_summary['frequency'], rfm_summary['recency'], rfm_summary['T'])
    
    # Predict expected purchases in next 30/90 days
    rfm_summary['pred_purchases_30d'] = bgf.predict(30, rfm_summary['frequency'], 
                                                       rfm_summary['recency'], rfm_summary['T'])
    rfm_summary['pred_purchases_90d'] = bgf.predict(90, rfm_summary['frequency'],
                                                       rfm_summary['recency'], rfm_summary['T'])
    
    print(f"  BG/NBD avg predicted purchases (30d): {rfm_summary['pred_purchases_30d'].mean():.2f}")
    print(f"  BG/NBD avg predicted purchases (90d): {rfm_summary['pred_purchases_90d'].mean():.2f}")
    
    # Gamma-Gamma model for monetary value
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(rfm_summary['frequency'], rfm_summary['monetary_value'])
    
    rfm_summary['predicted_clv_90d'] = ggf.customer_lifetime_value(
        bgf, rfm_summary['frequency'], rfm_summary['recency'],
        rfm_summary['T'], rfm_summary['monetary_value'],
        time=3, discount_rate=0.01
    )
    
    print(f"  Gamma-Gamma avg CLV (90d): ${rfm_summary['predicted_clv_90d'].mean():.2f}")
    print(f"  Gamma-Gamma median CLV (90d): ${rfm_summary['predicted_clv_90d'].median():.2f}")
    
    bgnbd_results = {
        'n_customers': len(rfm_summary),
        'avg_pred_purchases_30d': round(float(rfm_summary['pred_purchases_30d'].mean()), 3),
        'avg_pred_purchases_90d': round(float(rfm_summary['pred_purchases_90d'].mean()), 3),
        'avg_clv_90d': round(float(rfm_summary['predicted_clv_90d'].mean()), 2),
        'median_clv_90d': round(float(rfm_summary['predicted_clv_90d'].median()), 2),
    }
    
except Exception as e:
    print(f"  BG/NBD model error (using fallback): {e}")
    # Fallback: simple frequency-based CLV estimate
    payer_rfm = payers[['player_id', 'frequency', 'monetary', 'recency']].copy()
    payer_rfm['avg_purchase_value'] = payer_rfm['monetary'] / payer_rfm['frequency'].clip(lower=1)
    payer_rfm['purchase_rate_per_day'] = payer_rfm['frequency'] / payer_rfm['recency'].clip(lower=1)
    payer_rfm['clv_90d_simple'] = payer_rfm['avg_purchase_value'] * payer_rfm['purchase_rate_per_day'] * 90
    
    print(f"  Fallback CLV avg (90d): ${payer_rfm['clv_90d_simple'].mean():.2f}")
    print(f"  Fallback CLV median (90d): ${payer_rfm['clv_90d_simple'].median():.2f}")
    
    bgnbd_results = {
        'method': 'simple_frequency',
        'avg_clv_90d': round(float(payer_rfm['clv_90d_simple'].mean()), 2),
        'median_clv_90d': round(float(payer_rfm['clv_90d_simple'].median()), 2),
    }

# ── Save all results ──
results = {
    'cox_ph': {
        'concordance_index': round(float(cph.concordance_index_), 4),
        'top_features': summary[['coef', 'exp(coef)', 'p']].sort_values('p').head(8).reset_index().to_dict(orient='records'),
    },
    'kaplan_meier': km_data,
    'xgboost_ltv': xgb_results,
    'bgnbd_clv': bgnbd_results,
}

with open(f"{PROCESSED}/ltv_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✓ Module 4 complete!")
