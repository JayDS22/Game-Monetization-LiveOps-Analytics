"""
Module 2: Player Segmentation & Whale Detection
=================================================
- K-Means + DBSCAN clustering on behavioral+spending features
- RFM segmentation
- Silhouette analysis, cluster stability
- SHAP feature importance per segment
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import json, os, warnings
warnings.filterwarnings('ignore')

PROCESSED = "/home/claude/game-monetization/data/processed"

print("=" * 60)
print("MODULE 2: PLAYER SEGMENTATION & WHALE DETECTION")
print("=" * 60)

df = pd.read_csv(f"{PROCESSED}/master_features.csv")
print(f"Loaded {len(df):,} players with {len(df.columns)} features")

# ── Segmentation Features ──
seg_features = [
    'total_sessions', 'avg_session_minutes', 'total_play_hours',
    'days_active', 'max_stage_reached', 'session_frequency',
    'play_intensity', 'engagement_ratio', 'max_play_streak',
    'total_spend_usd', 'num_purchases', 'is_payer',
    'd1_retained', 'd7_retained', 'd30_retained',
    'fb_connected', 'is_ios', 'age'
]

X_seg = df[seg_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_seg)

# ── Subsample for clustering (memory efficient) ──
SAMPLE_N = 50000
np.random.seed(42)
sample_idx = np.random.choice(len(df), SAMPLE_N, replace=False)
X_sample = X_scaled[sample_idx]

# ── K-Means with Silhouette Analysis ──
print("\n--- K-Means Clustering (on 50K subsample) ---")
silhouette_scores = {}
inertias = {}

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_sample)
    sil = silhouette_score(X_sample, labels, sample_size=5000)
    silhouette_scores[k] = round(sil, 4)
    inertias[k] = round(km.inertia_, 2)
    print(f"  k={k}: Silhouette={sil:.4f}, Inertia={km.inertia_:.0f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nBest k={best_k} (silhouette={silhouette_scores[best_k]:.4f})")

# Final K-Means on full data
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['kmeans_cluster'] = km_final.fit_predict(X_scaled)
del X_scaled  # free memory

# Cluster profiles
print(f"\n--- K-Means Cluster Profiles (k={best_k}) ---")
cluster_profiles = df.groupby('kmeans_cluster').agg(
    count=('player_id', 'count'),
    avg_sessions=('total_sessions', 'mean'),
    avg_spend=('total_spend_usd', 'mean'),
    payer_rate=('is_payer', 'mean'),
    avg_retention_d7=('d7_retained', 'mean'),
    avg_stage=('max_stage_reached', 'mean'),
    avg_engagement=('engagement_ratio', 'mean'),
).round(3)
print(cluster_profiles)

# ── DBSCAN for outlier/whale detection (payers only for memory) ──
print("\n--- DBSCAN (Whale/Outlier Detection on payers) ---")
payer_mask_idx = df[df['is_payer'] == 1].index
dbscan_features = ['total_spend_usd', 'num_purchases', 'total_sessions', 'total_play_hours', 'max_stage_reached']
X_dbscan = StandardScaler().fit_transform(df.loc[payer_mask_idx, dbscan_features].fillna(0))

dbscan = DBSCAN(eps=1.5, min_samples=5)
db_labels = dbscan.fit_predict(X_dbscan)
df['dbscan_cluster'] = -2  # default for non-payers
df.loc[payer_mask_idx, 'dbscan_cluster'] = db_labels
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_outliers = (db_labels == -1).sum()
print(f"  DBSCAN clusters: {n_clusters_db}, Outliers (potential whales): {n_outliers:,}")
del X_dbscan

# ── RFM Segmentation (payers only) ──
print("\n--- RFM Segmentation ---")
payers = df[df['is_payer'] == 1].copy()

def rfm_segment(row):
    r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 4:
        return 'Loyal Whales'
    elif r >= 4 and f <= 2:
        return 'New Payers'
    elif r <= 2 and f >= 3:
        return 'At Risk High Value'
    elif r <= 2 and f <= 2 and m >= 3:
        return 'Lapsed Whales'
    elif r >= 3 and m <= 2:
        return 'Low Spend Active'
    elif r <= 2:
        return 'Churning'
    else:
        return 'Potential Loyalists'

payers['rfm_segment'] = payers.apply(rfm_segment, axis=1)
rfm_dist = payers['rfm_segment'].value_counts()
print(rfm_dist)

# Merge back
df = df.merge(payers[['player_id', 'rfm_segment']], on='player_id', how='left')
df['rfm_segment'] = df['rfm_segment'].fillna('Free Player')

# ── Player Tier Revenue Concentration ──
print("\n--- Revenue Concentration by Tier ---")
tier_revenue = df.groupby('player_tier').agg(
    n_players=('player_id', 'count'),
    total_rev=('total_spend_usd', 'sum'),
    avg_rev=('total_spend_usd', 'mean'),
).reset_index()
tier_revenue['pct_players'] = (tier_revenue['n_players'] / tier_revenue['n_players'].sum() * 100).round(2)
tier_revenue['pct_revenue'] = (tier_revenue['total_rev'] / tier_revenue['total_rev'].sum() * 100).round(2)
print(tier_revenue)

# ── PCA for visualization (subsample) ──
print("\nComputing PCA projection (50K subsample)...")
X_pca_input = scaler.transform(df.loc[sample_idx, seg_features].fillna(0))
pca = PCA(n_components=2, random_state=42)
pca.fit(X_pca_input)
# Transform a small sample for storage
df['pca_1'] = 0.0
df['pca_2'] = 0.0
X_pca_all = scaler.transform(df[seg_features].fillna(0))
X_pca_proj = pca.transform(X_pca_all)
df['pca_1'] = X_pca_proj[:, 0]
df['pca_2'] = X_pca_proj[:, 1]
del X_pca_input, X_pca_all, X_pca_proj
print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# ── Save results ──
df.to_csv(f"{PROCESSED}/segmented_players.csv", index=False)

# Save cluster profiles for dashboard
results = {
    'silhouette_scores': silhouette_scores,
    'inertias': {str(k): v for k, v in inertias.items()},
    'best_k': best_k,
    'cluster_profiles': cluster_profiles.reset_index().to_dict(orient='records'),
    'tier_revenue': tier_revenue.to_dict(orient='records'),
    'rfm_distribution': rfm_dist.to_dict(),
    'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
}
with open(f"{PROCESSED}/segmentation_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✓ Module 2 complete! Saved segmented_players.csv")
