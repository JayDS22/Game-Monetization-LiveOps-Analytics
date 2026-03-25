"""
Module 3: Conversion Funnel Analysis
======================================
- F2P-to-payer journey mapping
- Drop-off rates at each stage
- Time-to-convert distributions
- Platform/country/gender effects (logistic regression)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import json, os, warnings
warnings.filterwarnings('ignore')

PROCESSED = "/home/claude/game-monetization/data/processed"

print("=" * 60)
print("MODULE 3: CONVERSION FUNNEL ANALYSIS")
print("=" * 60)

df = pd.read_csv(f"{PROCESSED}/segmented_players.csv")
print(f"Loaded {len(df):,} players")

# ── Funnel Stages ──
funnel = {
    'Install': len(df),
    'Tutorial Complete': df['tutorial_completed'].sum(),
    'D1 Retained': df['d1_retained'].sum(),
    'D7 Retained': df['d7_retained'].sum(),
    'D30 Retained': df['d30_retained'].sum(),
    'First IAP': df['is_payer'].sum(),
    'Repeat Purchaser (2+)': (df['num_purchases'] >= 2).sum(),
    'High Value (5+ purchases)': (df['num_purchases'] >= 5).sum(),
}

print("\n--- Conversion Funnel ---")
prev = None
funnel_data = []
for stage, count in funnel.items():
    pct_total = count / funnel['Install'] * 100
    pct_prev = count / prev * 100 if prev else 100.0
    drop = 100 - pct_prev
    funnel_data.append({
        'stage': stage, 'count': int(count),
        'pct_of_install': round(pct_total, 2),
        'pct_of_previous': round(pct_prev, 2),
        'dropoff_rate': round(drop, 2) if prev else 0.0
    })
    print(f"  {stage:30s}: {count:>8,} ({pct_total:5.1f}% of install, {drop:5.1f}% drop)")
    prev = count

# ── Time-to-Convert Distribution ──
print("\n--- Time-to-First-Purchase Distribution ---")
payers = df[df['is_payer'] == 1].copy()
ttfp = payers['days_to_first_purchase'].dropna()

print(f"  Median: {ttfp.median():.0f} days")
print(f"  Mean: {ttfp.mean():.1f} days")
print(f"  P25: {ttfp.quantile(0.25):.0f} days")
print(f"  P75: {ttfp.quantile(0.75):.0f} days")
print(f"  P90: {ttfp.quantile(0.90):.0f} days")

# Time buckets
time_buckets = pd.cut(ttfp, bins=[0, 1, 3, 7, 14, 30, 90, 365], 
                       labels=['Day 0-1', 'Day 2-3', 'Day 4-7', 'Day 8-14', 'Day 15-30', 'Day 31-90', 'Day 91+'])
time_dist = time_buckets.value_counts().sort_index()
print(f"\n  Time-to-convert distribution:")
for bucket, count in time_dist.items():
    print(f"    {bucket}: {count:,} ({count/len(ttfp)*100:.1f}%)")

# ── Friction Point Analysis ──
print("\n--- Friction Point Analysis (Chi-squared tests) ---")

# Test: does tutorial completion affect conversion?
ct_tutorial = pd.crosstab(df['tutorial_completed'], df['is_payer'])
chi2, p_tut, _, _ = stats.chi2_contingency(ct_tutorial)
conv_tut = df.groupby('tutorial_completed')['is_payer'].mean()
print(f"  Tutorial -> Conversion: completed={conv_tut.get(1, 0):.3f} vs not={conv_tut.get(0, 0):.3f} (p={p_tut:.2e})")

# Test: does D1 retention affect conversion?
ct_d1 = pd.crosstab(df['d1_retained'], df['is_payer'])
chi2, p_d1, _, _ = stats.chi2_contingency(ct_d1)
conv_d1 = df.groupby('d1_retained')['is_payer'].mean()
print(f"  D1 Retained -> Conversion: yes={conv_d1.get(1, 0):.3f} vs no={conv_d1.get(0, 0):.3f} (p={p_d1:.2e})")

# Test: FB connected effect
ct_fb = pd.crosstab(df['fb_connected'], df['is_payer'])
chi2, p_fb, _, _ = stats.chi2_contingency(ct_fb)
conv_fb = df.groupby('fb_connected')['is_payer'].mean()
print(f"  FB Connected -> Conversion: yes={conv_fb.get(1, 0):.3f} vs no={conv_fb.get(0, 0):.3f} (p={p_fb:.2e})")

# ── Platform/Country/Gender Effects (Logistic Regression) ──
print("\n--- Logistic Regression: Conversion Drivers ---")

# Prepare features
lr_df = df[['is_payer', 'platform', 'country', 'gender', 'age', 'tutorial_completed',
            'd1_retained', 'fb_connected', 'total_sessions', 'max_stage_reached']].copy()

# Encode categoricals
le_platform = LabelEncoder()
lr_df['platform_enc'] = le_platform.fit_transform(lr_df['platform'])
le_gender = LabelEncoder()
lr_df['gender_enc'] = le_gender.fit_transform(lr_df['gender'])

# Top countries as dummies
top_countries = df['country'].value_counts().head(8).index
for c in top_countries:
    lr_df[f'country_{c}'] = (lr_df['country'] == c).astype(int)

feature_cols = ['platform_enc', 'gender_enc', 'age', 'tutorial_completed', 
                'd1_retained', 'fb_connected', 'total_sessions', 'max_stage_reached'] + \
               [f'country_{c}' for c in top_countries]

X_lr = lr_df[feature_cols].fillna(0)
y_lr = lr_df['is_payer']

lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_lr, y_lr)

print(f"  Accuracy: {lr.score(X_lr, y_lr):.3f}")
print(f"\n  Feature Coefficients (odds ratios):")
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr.coef_[0],
    'odds_ratio': np.exp(lr.coef_[0])
}).sort_values('coefficient', ascending=False)

for _, row in coef_df.iterrows():
    direction = "↑" if row['coefficient'] > 0 else "↓"
    print(f"    {direction} {row['feature']:25s}: coef={row['coefficient']:+.4f}, OR={row['odds_ratio']:.3f}")

# ── Platform-specific conversion rates ──
print("\n--- Conversion by Segment ---")
for col, name in [('platform', 'Platform'), ('gender', 'Gender')]:
    conv = df.groupby(col)['is_payer'].agg(['mean', 'count']).round(4)
    conv.columns = ['conversion_rate', 'n_players']
    print(f"\n  {name}:")
    for idx, row in conv.iterrows():
        print(f"    {idx}: {row['conversion_rate']:.3%} (n={row['n_players']:,})")

# Country conversion
print("\n  Top Countries by Conversion:")
country_conv = df.groupby('country')['is_payer'].agg(['mean', 'count']).sort_values('mean', ascending=False)
country_conv.columns = ['conversion_rate', 'n_players']
for idx, row in country_conv.head(10).iterrows():
    print(f"    {idx}: {row['conversion_rate']:.3%} (n={row['n_players']:,})")

# ── Save results ──
results = {
    'funnel': funnel_data,
    'time_to_convert': {
        'median_days': float(ttfp.median()),
        'mean_days': float(ttfp.mean()),
        'p25_days': float(ttfp.quantile(0.25)),
        'p75_days': float(ttfp.quantile(0.75)),
        'distribution': {str(k): int(v) for k, v in time_dist.items()},
    },
    'friction_tests': {
        'tutorial_p': float(p_tut),
        'd1_retention_p': float(p_d1),
        'fb_connected_p': float(p_fb),
    },
    'logistic_regression': {
        'accuracy': float(lr.score(X_lr, y_lr)),
        'coefficients': coef_df.to_dict(orient='records'),
    },
    'platform_conversion': df.groupby('platform')['is_payer'].mean().to_dict(),
    'country_conversion': country_conv['conversion_rate'].head(10).to_dict(),
}

with open(f"{PROCESSED}/funnel_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✓ Module 3 complete!")
