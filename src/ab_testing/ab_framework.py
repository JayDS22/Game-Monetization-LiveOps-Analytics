"""
Module 5: A/B Testing & Experimentation Framework
====================================================
- Cookie Cats gate placement analysis
- Reusable A/B testing module: z-test, chi-squared, bootstrap CI
- Power analysis & MDE estimation
- Pricing A/B test simulation with CUPED variance reduction
- Bayesian A/B testing
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import json, os, warnings
warnings.filterwarnings('ignore')

RAW = "/home/claude/game-monetization/data/raw"
PROCESSED = "/home/claude/game-monetization/data/processed"

print("=" * 60)
print("MODULE 5: A/B TESTING & EXPERIMENTATION FRAMEWORK")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# 1. COOKIE CATS GATE PLACEMENT EXPERIMENT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("1. COOKIE CATS GATE PLACEMENT A/B TEST")
print("=" * 50)

ab = pd.read_csv(f"{RAW}/cookie_cats_ab.csv")
print(f"Loaded {len(ab):,} users")
print(f"Groups: {ab['version'].value_counts().to_dict()}")

gate30 = ab[ab['version'] == 'gate_30']
gate40 = ab[ab['version'] == 'gate_40']

# ── Z-test for proportions (D1 retention) ──
def z_test_proportions(group_a, group_b, col):
    n_a, n_b = len(group_a), len(group_b)
    p_a, p_b = group_a[col].mean(), group_b[col].mean()
    p_pool = (group_a[col].sum() + group_b[col].sum()) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_a - p_b) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {'p_a': p_a, 'p_b': p_b, 'diff': p_a - p_b, 'relative_lift': (p_a - p_b) / p_b,
            'z_stat': z, 'p_value': p_value, 'se': se, 'significant': p_value < 0.05}

print("\n--- D1 Retention ---")
d1_result = z_test_proportions(gate30, gate40, 'retention_1')
print(f"  Gate 30: {d1_result['p_a']:.4f}")
print(f"  Gate 40: {d1_result['p_b']:.4f}")
print(f"  Difference: {d1_result['diff']:+.4f} ({d1_result['relative_lift']:+.2%})")
print(f"  Z-stat: {d1_result['z_stat']:.3f}, p-value: {d1_result['p_value']:.4f}")
print(f"  Significant: {d1_result['significant']}")

print("\n--- D7 Retention ---")
d7_result = z_test_proportions(gate30, gate40, 'retention_7')
print(f"  Gate 30: {d7_result['p_a']:.4f}")
print(f"  Gate 40: {d7_result['p_b']:.4f}")
print(f"  Difference: {d7_result['diff']:+.4f} ({d7_result['relative_lift']:+.2%})")
print(f"  Z-stat: {d7_result['z_stat']:.3f}, p-value: {d7_result['p_value']:.4f}")
print(f"  Significant: {d7_result['significant']}")

# ── Chi-squared test ──
print("\n--- Chi-Squared Tests ---")
for ret_col in ['retention_1', 'retention_7']:
    ct = pd.crosstab(ab['version'], ab[ret_col])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"  {ret_col}: χ²={chi2:.3f}, p={p:.4f}, significant={p < 0.05}")

# ── Bootstrap Confidence Intervals ──
print("\n--- Bootstrap 95% CI for D7 Retention Difference ---")
np.random.seed(42)
n_bootstrap = 10000
boot_diffs = []
for _ in range(n_bootstrap):
    s30 = gate30['retention_7'].sample(n=len(gate30), replace=True).mean()
    s40 = gate40['retention_7'].sample(n=len(gate40), replace=True).mean()
    boot_diffs.append(s30 - s40)

ci_lower = np.percentile(boot_diffs, 2.5)
ci_upper = np.percentile(boot_diffs, 97.5)
print(f"  Bootstrap mean diff: {np.mean(boot_diffs):.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  CI excludes 0: {ci_lower > 0 or ci_upper < 0}")

# ── Engagement effect (game rounds) ──
print("\n--- Game Rounds Comparison ---")
t_stat, p_rounds = stats.ttest_ind(gate30['sum_gamerounds'], gate40['sum_gamerounds'])
print(f"  Gate 30 mean: {gate30['sum_gamerounds'].mean():.1f}")
print(f"  Gate 40 mean: {gate40['sum_gamerounds'].mean():.1f}")
print(f"  t-stat: {t_stat:.3f}, p-value: {p_rounds:.4f}")

# ══════════════════════════════════════════════════════════════
# 2. POWER ANALYSIS & MDE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("2. POWER ANALYSIS CALCULATOR")
print("=" * 50)

def sample_size_calculator(baseline_rate, mde, alpha=0.05, power=0.80):
    """Calculate required sample size per group for proportion test."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + mde
    n = ((z_alpha * np.sqrt(2 * p1 * (1 - p1)) + 
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / mde) ** 2
    return int(np.ceil(n))

# Power analysis for various scenarios
scenarios = [
    ("D1 Retention (baseline=52%)", 0.52, [0.01, 0.02, 0.03, 0.05]),
    ("D7 Retention (baseline=21%)", 0.21, [0.01, 0.02, 0.03, 0.05]),
    ("Conversion Rate (baseline=3.2%)", 0.032, [0.005, 0.01, 0.015, 0.02]),
]

power_results = []
for name, baseline, mdes in scenarios:
    print(f"\n  {name}:")
    for mde in mdes:
        n = sample_size_calculator(baseline, mde)
        print(f"    MDE={mde:.3f}: n={n:,} per group ({2*n:,} total)")
        power_results.append({'scenario': name, 'baseline': baseline, 'mde': mde, 'n_per_group': n})

# MDE calculator (given sample size)
def mde_calculator(n, baseline_rate, alpha=0.05, power=0.80):
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n)
    mde = (z_alpha + z_beta) * se
    return mde

print(f"\n  MDE with n=45,000/group (Cookie Cats size):")
for name, baseline, _ in scenarios:
    mde = mde_calculator(45000, baseline)
    print(f"    {name}: MDE = {mde:.4f} ({mde/baseline:.1%} relative)")

# ══════════════════════════════════════════════════════════════
# 3. PRICING A/B TEST WITH CUPED
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("3. PRICING A/B TEST SIMULATION WITH CUPED")
print("=" * 50)

# Simulate pricing experiment: 20% discount on IAP bundles
np.random.seed(42)
n_exp = 20000

# Pre-experiment covariate (sessions in week before experiment)
pre_sessions = np.random.poisson(5, n_exp)
# Revenue is correlated with pre-experiment sessions
base_revenue = 0.5 + 0.3 * pre_sessions + np.random.exponential(0.5, n_exp)

# Treatment: 20% discount -> 8% lift in revenue (true effect)
treatment = np.random.binomial(1, 0.5, n_exp)
true_effect = 0.08
post_revenue = base_revenue * (1 + true_effect * treatment) + np.random.normal(0, 0.5, n_exp)
post_revenue = np.clip(post_revenue, 0, None)

exp_df = pd.DataFrame({
    'pre_sessions': pre_sessions,
    'treatment': treatment,
    'revenue': post_revenue,
})

# Standard t-test (no CUPED)
control_rev = exp_df[exp_df['treatment'] == 0]['revenue']
treatment_rev = exp_df[exp_df['treatment'] == 1]['revenue']
t_raw, p_raw = stats.ttest_ind(treatment_rev, control_rev)
raw_diff = treatment_rev.mean() - control_rev.mean()
raw_se = np.sqrt(treatment_rev.var()/len(treatment_rev) + control_rev.var()/len(control_rev))

print(f"\n  Standard t-test:")
print(f"    Control mean: ${control_rev.mean():.3f}")
print(f"    Treatment mean: ${treatment_rev.mean():.3f}")
print(f"    Diff: ${raw_diff:.4f}, SE: {raw_se:.4f}")
print(f"    t={t_raw:.3f}, p={p_raw:.4f}")

# CUPED variance reduction
theta = np.cov(exp_df['revenue'], exp_df['pre_sessions'])[0, 1] / np.var(exp_df['pre_sessions'])
exp_df['revenue_cuped'] = exp_df['revenue'] - theta * (exp_df['pre_sessions'] - exp_df['pre_sessions'].mean())

control_cuped = exp_df[exp_df['treatment'] == 0]['revenue_cuped']
treatment_cuped = exp_df[exp_df['treatment'] == 1]['revenue_cuped']
t_cuped, p_cuped = stats.ttest_ind(treatment_cuped, control_cuped)
cuped_diff = treatment_cuped.mean() - control_cuped.mean()
cuped_se = np.sqrt(treatment_cuped.var()/len(treatment_cuped) + control_cuped.var()/len(control_cuped))

variance_reduction = 1 - (exp_df['revenue_cuped'].var() / exp_df['revenue'].var())

print(f"\n  CUPED-adjusted t-test:")
print(f"    Control mean: ${control_cuped.mean():.3f}")
print(f"    Treatment mean: ${treatment_cuped.mean():.3f}")
print(f"    Diff: ${cuped_diff:.4f}, SE: {cuped_se:.4f}")
print(f"    t={t_cuped:.3f}, p={p_cuped:.4f}")
print(f"    Variance reduction: {variance_reduction:.1%}")

# ══════════════════════════════════════════════════════════════
# 4. BAYESIAN A/B TESTING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("4. BAYESIAN A/B TESTING")
print("=" * 50)

# D7 retention: gate_30 vs gate_40
successes_a = gate30['retention_7'].sum()
trials_a = len(gate30)
successes_b = gate40['retention_7'].sum()
trials_b = len(gate40)

# Beta posterior (uniform prior)
n_samples = 100000
posterior_a = np.random.beta(successes_a + 1, trials_a - successes_a + 1, n_samples)
posterior_b = np.random.beta(successes_b + 1, trials_b - successes_b + 1, n_samples)

prob_a_better = (posterior_a > posterior_b).mean()
expected_loss_a = np.mean(np.maximum(posterior_b - posterior_a, 0))
expected_loss_b = np.mean(np.maximum(posterior_a - posterior_b, 0))
credible_interval = np.percentile(posterior_a - posterior_b, [2.5, 97.5])

print(f"  P(Gate 30 > Gate 40): {prob_a_better:.3f}")
print(f"  Expected loss (choosing A): {expected_loss_a:.5f}")
print(f"  Expected loss (choosing B): {expected_loss_b:.5f}")
print(f"  95% credible interval for diff: [{credible_interval[0]:.4f}, {credible_interval[1]:.4f}]")
print(f"  Recommendation: {'Gate 30' if prob_a_better > 0.5 else 'Gate 40'} (P={max(prob_a_better, 1-prob_a_better):.3f})")

# ── Save all results ──
results = {
    'cookie_cats': {
        'd1_retention': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in d1_result.items()},
        'd7_retention': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in d7_result.items()},
        'bootstrap_ci': {'lower': float(ci_lower), 'upper': float(ci_upper), 'mean_diff': float(np.mean(boot_diffs))},
    },
    'power_analysis': power_results,
    'cuped_experiment': {
        'raw': {'diff': float(raw_diff), 'se': float(raw_se), 'p_value': float(p_raw)},
        'cuped': {'diff': float(cuped_diff), 'se': float(cuped_se), 'p_value': float(p_cuped)},
        'variance_reduction': float(variance_reduction),
        'true_effect': true_effect,
    },
    'bayesian': {
        'prob_a_better': float(prob_a_better),
        'expected_loss_a': float(expected_loss_a),
        'expected_loss_b': float(expected_loss_b),
        'credible_interval': [float(credible_interval[0]), float(credible_interval[1])],
    }
}

with open(f"{PROCESSED}/ab_testing_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, (np.bool_,)) else str(x))

print(f"\n✓ Module 5 complete!")
