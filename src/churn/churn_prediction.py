"""
Module 6: Churn Prediction & Live-Ops Recommendations
=======================================================
- Binary churn classifier (XGBoost, LightGBM) with D7 churn target
- SHAP explainability (waterfall + summary)
- Actionable live-ops recommendations
- Automated alert thresholds
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, 
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)
import xgboost as xgb
import lightgbm as lgb
import shap
import json, os, warnings
warnings.filterwarnings('ignore')

PROCESSED = "/home/claude/game-monetization/data/processed"

print("=" * 60)
print("MODULE 6: CHURN PREDICTION & LIVE-OPS")
print("=" * 60)

df = pd.read_csv(f"{PROCESSED}/segmented_players.csv")
print(f"Loaded {len(df):,} players")
print(f"Churn rate: {df['churned_d7'].mean():.1%}")

# ── Feature Selection ──
churn_features = [
    'total_sessions', 'avg_session_minutes', 'total_play_hours',
    'days_active', 'max_stage_reached', 'session_frequency',
    'play_intensity', 'engagement_ratio', 'max_play_streak',
    'stage_velocity', 'stage_per_session', 'stage_per_hour',
    'is_payer', 'total_spend_usd', 'num_purchases',
    'tutorial_completed', 'fb_connected', 'is_ios',
    'is_high_spend_country', 'age', 'retention_score',
    'd1_retained', 'd7_retained', 'd30_retained',
    'days_since_install', 'days_since_last_active',
]
churn_features = [f for f in churn_features if f in df.columns]

X = df[churn_features].fillna(0)
y = df['churned_d7']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")

# ══════════════════════════════════════════════════════════════
# 1. XGBOOST CHURN MODEL
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("1. XGBOOST CHURN CLASSIFIER")
print("=" * 50)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=42, tree_method='hist', eval_metric='auc'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_prob_xgb)
ap_xgb = average_precision_score(y_test, y_prob_xgb)

print(f"\n  ROC-AUC: {auc_xgb:.4f}")
print(f"  Average Precision: {ap_xgb:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Retained', 'Churned']))

# ══════════════════════════════════════════════════════════════
# 2. LIGHTGBM CHURN MODEL
# ══════════════════════════════════════════════════════════════
print("=" * 50)
print("2. LIGHTGBM CHURN CLASSIFIER")
print("=" * 50)

lgb_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
auc_lgb = roc_auc_score(y_test, y_prob_lgb)
ap_lgb = average_precision_score(y_test, y_prob_lgb)

print(f"\n  ROC-AUC: {auc_lgb:.4f}")
print(f"  Average Precision: {ap_lgb:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_lgb, target_names=['Retained', 'Churned']))

# Model comparison
print("\n--- Model Comparison ---")
print(f"  XGBoost  AUC: {auc_xgb:.4f} | AP: {ap_xgb:.4f}")
print(f"  LightGBM AUC: {auc_lgb:.4f} | AP: {ap_lgb:.4f}")
best_model_name = 'XGBoost' if auc_xgb >= auc_lgb else 'LightGBM'
best_model = xgb_model if auc_xgb >= auc_lgb else lgb_model
best_probs = y_prob_xgb if auc_xgb >= auc_lgb else y_prob_lgb
print(f"  Best model: {best_model_name}")

# ══════════════════════════════════════════════════════════════
# 3. SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("3. SHAP EXPLAINABILITY")
print("=" * 50)

# Use tree explainer on XGBoost (faster)
explainer = shap.TreeExplainer(xgb_model)
# Use a small sample for SHAP (memory)
shap_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(shap_sample)

# Global feature importance
shap_importance = pd.DataFrame({
    'feature': churn_features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\n  Top SHAP Features (global importance):")
for _, row in shap_importance.head(10).iterrows():
    print(f"    {row['feature']:30s}: {row['mean_abs_shap']:.4f}")

# Example individual explanations (top 3 churners)
print("\n  Example: Top 3 predicted churners explanation")
high_risk = X_test.copy()
high_risk['churn_prob'] = best_probs
top_churners = high_risk.nlargest(3, 'churn_prob')

for i, (idx, row) in enumerate(top_churners.iterrows()):
    player_shap = explainer.shap_values(X_test.loc[[idx]])
    top_reasons = pd.Series(player_shap[0], index=churn_features).abs().nlargest(3)
    print(f"\n    Churner {i+1} (P={row['churn_prob']:.3f}):")
    for feat, val in top_reasons.items():
        direction = "↑" if player_shap[0][churn_features.index(feat)] > 0 else "↓"
        print(f"      {direction} {feat}: SHAP={val:.3f}, value={row[feat]:.1f}")

# ══════════════════════════════════════════════════════════════
# 4. LIVE-OPS RECOMMENDATIONS & ALERTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("4. LIVE-OPS RECOMMENDATIONS")
print("=" * 50)

# Score all players
df['churn_probability'] = best_model.predict_proba(X.fillna(0))[:, 1]

# Risk tiers
df['churn_risk'] = pd.cut(df['churn_probability'], bins=[0, 0.3, 0.6, 0.8, 1.0],
                           labels=['Low', 'Medium', 'High', 'Critical'])

risk_dist = df['churn_risk'].value_counts()
print(f"\n  Churn Risk Distribution:")
for risk, count in risk_dist.items():
    print(f"    {risk}: {count:,} ({count/len(df)*100:.1f}%)")

# Targeted recommendations
recommendations = [
    {
        'segment': 'High-value at-risk (whales/dolphins, churn_prob > 0.6)',
        'count': int(len(df[(df['player_tier'].isin(['whale', 'dolphin'])) & (df['churn_probability'] > 0.6)])),
        'action': 'VIP personal outreach + exclusive limited-time offers',
        'priority': 'CRITICAL',
    },
    {
        'segment': 'Inactive 3-5 days',
        'count': int(len(df[(df['days_since_last_active'] >= 3) & (df['days_since_last_active'] < 6)])),
        'action': 'Push notification with comeback bonus (free gems/energy)',
        'priority': 'HIGH',
    },
    {
        'segment': 'Tutorial dropoffs (no tutorial completion, sessions > 0)',
        'count': int(len(df[(df['tutorial_completed'] == 0) & (df['total_sessions'] > 1)])),
        'action': 'Simplified onboarding flow + tutorial skip option',
        'priority': 'MEDIUM',
    },
    {
        'segment': 'Minnows at risk of churning (low spend, high churn prob)',
        'count': int(len(df[(df['player_tier'] == 'minnow') & (df['churn_probability'] > 0.7)])),
        'action': 'Targeted discount: 30% off first IAP bundle > $4.99',
        'priority': 'HIGH',
    },
    {
        'segment': 'Engaged non-payers (high sessions, zero spend)',
        'count': int(len(df[(df['total_sessions'] > 50) & (df['is_payer'] == 0)])),
        'action': 'Starter pack popup at session milestone (limited offer)',
        'priority': 'MEDIUM',
    },
]

print("\n  Actionable Recommendations:")
for rec in recommendations:
    print(f"\n    [{rec['priority']}] {rec['segment']}")
    print(f"    Players: {rec['count']:,}")
    print(f"    Action: {rec['action']}")

# ── Alert thresholds ──
print("\n--- Automated Alert Thresholds ---")
alerts = {
    'dau_drop_pct': 10,  # Alert if DAU drops >10% week-over-week
    'd1_retention_min': 0.35,  # Alert if D1 < 35%
    'd7_retention_min': 0.12,  # Alert if D7 < 12%
    'arpu_drop_pct': 15,  # Alert if ARPU drops >15%
    'conversion_rate_min': 0.02,  # Alert if conversion < 2%
    'whale_churn_count': 5,  # Alert if >5 whales at critical risk
}
print(f"  Configured alert thresholds: {json.dumps(alerts, indent=4)}")

# Current metric values
current_metrics = {
    'dau_estimate': int(df[df['days_since_last_active'] <= 1]['player_id'].count()),
    'mau_estimate': int(df[df['days_since_last_active'] <= 30]['player_id'].count()),
    'd1_retention': round(float(df['d1_retained'].mean()), 4),
    'd7_retention': round(float(df['d7_retained'].mean()), 4),
    'd30_retention': round(float(df['d30_retained'].mean()), 4),
    'arpu': round(float(df['total_spend_usd'].sum() / len(df)), 4),
    'arppu': round(float(df[df['is_payer']==1]['total_spend_usd'].mean()), 2),
    'conversion_rate': round(float(df['is_payer'].mean()), 4),
    'whale_pct': round(float((df['player_tier'] == 'whale').mean() * 100), 3),
    'whales_at_critical_risk': int(len(df[(df['player_tier'] == 'whale') & (df['churn_probability'] > 0.8)])),
}
print(f"\n  Current Metrics: {json.dumps(current_metrics, indent=4)}")

# ── Save results ──
# Save scored players
df[['player_id', 'churn_probability', 'churn_risk']].to_csv(f"{PROCESSED}/churn_scores.csv", index=False)

results = {
    'model_comparison': {
        'xgboost': {'auc': round(auc_xgb, 4), 'ap': round(ap_xgb, 4)},
        'lightgbm': {'auc': round(auc_lgb, 4), 'ap': round(ap_lgb, 4)},
        'best': best_model_name,
    },
    'shap_importance': shap_importance.head(15).to_dict(orient='records'),
    'risk_distribution': {str(k): int(v) for k, v in risk_dist.items()},
    'recommendations': recommendations,
    'alert_thresholds': alerts,
    'current_metrics': current_metrics,
    'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist(),
}

with open(f"{PROCESSED}/churn_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✓ Module 6 complete!")
