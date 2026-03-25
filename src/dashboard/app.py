"""
Game Monetization & Live-Ops Analytics Dashboard
==================================================
Streamlit multi-tab dashboard for F2P game analytics
Tabs: Executive Summary | Player Segments | Funnels | LTV | A/B Tests | Churn Alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

# ── Page Config ──
st.set_page_config(
    page_title="Game Monetization & Live-Ops Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d2ff; }
    .metric-label { font-size: 0.85rem; color: #a0a0a0; text-transform: uppercase; }
    .metric-delta { font-size: 0.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        padding: 10px 20px; border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED = os.path.join(DATA_DIR, 'processed')
RAW = os.path.join(DATA_DIR, 'raw')

@st.cache_data
def load_data():
    players = pd.read_csv(os.path.join(PROCESSED, 'segmented_players.csv'))
    txns = pd.read_csv(os.path.join(RAW, 'transactions.csv'), parse_dates=['transaction_date'])
    ab = pd.read_csv(os.path.join(RAW, 'cookie_cats_ab.csv'))
    
    churn_scores = pd.read_csv(os.path.join(PROCESSED, 'churn_scores.csv'))
    players = players.merge(churn_scores, on='player_id', how='left')
    
    with open(os.path.join(PROCESSED, 'segmentation_results.json')) as f:
        seg_results = json.load(f)
    with open(os.path.join(PROCESSED, 'funnel_results.json')) as f:
        funnel_results = json.load(f)
    with open(os.path.join(PROCESSED, 'ltv_results.json')) as f:
        ltv_results = json.load(f)
    with open(os.path.join(PROCESSED, 'ab_testing_results.json')) as f:
        ab_results = json.load(f)
    with open(os.path.join(PROCESSED, 'churn_results.json')) as f:
        churn_results = json.load(f)
    
    return players, txns, ab, seg_results, funnel_results, ltv_results, ab_results, churn_results

players, txns, ab_data, seg_results, funnel_results, ltv_results, ab_results, churn_results = load_data()

# ── Sidebar ──
st.sidebar.image("https://img.icons8.com/fluency/96/controller.png", width=60)
st.sidebar.title("🎮 Game Analytics")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Players:** {len(players):,}")
st.sidebar.markdown(f"**Payers:** {players['is_payer'].sum():,} ({players['is_payer'].mean():.1%})")
st.sidebar.markdown(f"**Total Revenue:** ${players['total_spend_usd'].sum():,.0f}")
st.sidebar.markdown(f"**Transactions:** {len(txns):,}")
st.sidebar.markdown("---")

# Platform filter
platform_filter = st.sidebar.multiselect("Platform", players['platform'].unique(), default=players['platform'].unique())
filtered = players[players['platform'].isin(platform_filter)]

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary", "👥 Player Segments", "🔄 Conversion Funnels",
    "💰 LTV Forecasting", "🧪 A/B Testing", "🚨 Churn & Alerts"
])

# ══════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("Executive Summary")
    st.caption("Real-time KPIs for F2P game monetization and player health")
    
    metrics = churn_results.get('current_metrics', {})
    
    # KPI Cards Row 1
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("DAU", f"{metrics.get('dau_estimate', 0):,}", help="Daily Active Users")
    c2.metric("MAU", f"{metrics.get('mau_estimate', 0):,}", help="Monthly Active Users")
    c3.metric("DAU/MAU", f"{metrics.get('dau_estimate', 1)/max(metrics.get('mau_estimate', 1), 1):.1%}", help="Stickiness ratio")
    c4.metric("Conversion", f"{metrics.get('conversion_rate', 0):.1%}", help="Install to payer")
    c5.metric("Whale %", f"{metrics.get('whale_pct', 0):.2f}%", help="% of payers who are whales")
    
    # KPI Cards Row 2
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("D1 Retention", f"{metrics.get('d1_retention', 0):.1%}")
    c2.metric("D7 Retention", f"{metrics.get('d7_retention', 0):.1%}")
    c3.metric("D30 Retention", f"{metrics.get('d30_retention', 0):.1%}")
    c4.metric("ARPU", f"${metrics.get('arpu', 0):.2f}")
    c5.metric("ARPPU", f"${metrics.get('arppu', 0):.2f}")
    
    st.markdown("---")
    
    # Revenue breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Player Tier")
        tier_data = pd.DataFrame(seg_results['tier_revenue'])
        fig_tier = px.pie(tier_data[tier_data['total_rev'] > 0], 
                          values='total_rev', names='player_tier',
                          color_discrete_sequence=px.colors.sequential.Plasma_r,
                          hole=0.4)
        fig_tier.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_tier, use_container_width=True)
    
    with col2:
        st.subheader("Revenue Over Time")
        rev_daily = txns.groupby(txns['transaction_date'].dt.to_period('W').astype(str))['price_usd'].sum().reset_index()
        rev_daily.columns = ['week', 'revenue']
        fig_rev = px.bar(rev_daily, x='week', y='revenue', 
                         color_discrete_sequence=['#00d2ff'])
        fig_rev.update_layout(height=350, margin=dict(t=20, b=20), xaxis_title="Week", yaxis_title="Revenue ($)")
        fig_rev.update_xaxes(tickangle=45, nticks=20)
        st.plotly_chart(fig_rev, use_container_width=True)
    
    # Cohort Retention Heatmap
    st.subheader("Cohort Retention Heatmap")
    filtered['install_cohort'] = pd.to_datetime(filtered['install_date']).dt.to_period('M').astype(str)
    cohort_ret = filtered.groupby('install_cohort').agg(
        size=('player_id', 'count'),
        d1=('d1_retained', 'mean'),
        d7=('d7_retained', 'mean'),
        d30=('d30_retained', 'mean'),
    ).reset_index()
    cohort_ret = cohort_ret[cohort_ret['size'] >= 100]
    
    heatmap_data = cohort_ret[['install_cohort', 'd1', 'd7', 'd30']].set_index('install_cohort').T
    heatmap_data.index = ['D1', 'D7', 'D30']
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_data.values * 100,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlOrRd_r',
        text=np.round(heatmap_data.values * 100, 1),
        texttemplate='%{text:.1f}%',
        textfont={"size": 11},
        colorbar=dict(title="Retention %"),
    ))
    fig_heat.update_layout(height=250, margin=dict(t=10, b=10), xaxis_title="Install Cohort")
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2: PLAYER SEGMENTS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("Player Segmentation & Whale Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("K-Means Silhouette Analysis")
        sil_data = pd.DataFrame([
            {'k': int(k), 'silhouette': v} 
            for k, v in seg_results['silhouette_scores'].items()
        ])
        fig_sil = px.line(sil_data, x='k', y='silhouette', markers=True,
                          color_discrete_sequence=['#00d2ff'])
        fig_sil.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_sil, use_container_width=True)
    
    with col2:
        st.subheader("Revenue Concentration (Pareto)")
        tier_df = pd.DataFrame(seg_results['tier_revenue'])
        tier_df = tier_df[tier_df['total_rev'] > 0].sort_values('total_rev', ascending=False)
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=tier_df['player_tier'], y=tier_df['pct_revenue'], 
                                     name='% Revenue', marker_color='#ff6b6b'))
        fig_pareto.add_trace(go.Bar(x=tier_df['player_tier'], y=tier_df['pct_players'],
                                     name='% Players', marker_color='#4ecdc4'))
        fig_pareto.update_layout(barmode='group', height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pareto, use_container_width=True)
    
    # PCA Cluster Visualization
    st.subheader("PCA Cluster Visualization")
    sample_viz = filtered.sample(n=min(5000, len(filtered)), random_state=42)
    fig_pca = px.scatter(sample_viz, x='pca_1', y='pca_2', color='player_tier',
                         opacity=0.5, color_discrete_map={
                             'whale': '#ff0000', 'dolphin': '#ff9800',
                             'minnow': '#4caf50', 'free_rider': '#90a4ae'
                         })
    fig_pca.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # RFM Segment Distribution
    st.subheader("RFM Segment Distribution (Payers)")
    rfm_dist = pd.DataFrame([
        {'segment': k, 'count': v}
        for k, v in seg_results['rfm_distribution'].items()
    ]).sort_values('count', ascending=True)
    fig_rfm = px.bar(rfm_dist, y='segment', x='count', orientation='h',
                     color_discrete_sequence=['#7c4dff'])
    fig_rfm.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_rfm, use_container_width=True)
    
    # Cluster Profiles Table
    st.subheader("K-Means Cluster Profiles")
    st.dataframe(pd.DataFrame(seg_results['cluster_profiles']), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3: CONVERSION FUNNELS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("F2P Conversion Funnel Analysis")
    
    # Funnel Chart
    st.subheader("Install → First IAP → Repeat Purchaser Journey")
    funnel_df = pd.DataFrame(funnel_results['funnel'])
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_df['stage'],
        x=funnel_df['count'],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=['#667eea', '#764ba2', '#f093fb', '#f5576c', 
                           '#fda085', '#f6d365', '#96e6a1', '#72c6ef']),
    ))
    fig_funnel.update_layout(height=500, margin=dict(t=20, b=20))
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Drop-off Rates by Stage")
        drop_df = funnel_df[funnel_df['dropoff_rate'] > 0]
        fig_drop = px.bar(drop_df, x='stage', y='dropoff_rate',
                          color='dropoff_rate', color_continuous_scale='Reds',
                          text=drop_df['dropoff_rate'].apply(lambda x: f'{x:.1f}%'))
        fig_drop.update_layout(height=350, margin=dict(t=20, b=20))
        fig_drop.update_xaxes(tickangle=30)
        st.plotly_chart(fig_drop, use_container_width=True)
    
    with col2:
        st.subheader("Time-to-First-Purchase Distribution")
        ttc = funnel_results['time_to_convert']['distribution']
        ttc_df = pd.DataFrame([{'bucket': k, 'count': v} for k, v in ttc.items()])
        fig_ttc = px.bar(ttc_df, x='bucket', y='count', color_discrete_sequence=['#4ecdc4'],
                         text='count')
        fig_ttc.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_ttc, use_container_width=True)
    
    # Conversion by country
    st.subheader("Conversion Rate by Country")
    country_conv = funnel_results['country_conversion']
    cc_df = pd.DataFrame([{'country': k, 'rate': v} for k, v in country_conv.items()]).sort_values('rate', ascending=False)
    fig_cc = px.bar(cc_df, x='country', y='rate', color='rate', color_continuous_scale='Viridis',
                    text=cc_df['rate'].apply(lambda x: f'{x:.2%}'))
    fig_cc.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_cc, use_container_width=True)
    
    # Logistic regression coefficients
    st.subheader("Conversion Drivers (Logistic Regression)")
    coef_df = pd.DataFrame(funnel_results['logistic_regression']['coefficients'])
    coef_df = coef_df.sort_values('coefficient')
    fig_coef = px.bar(coef_df, y='feature', x='coefficient', orientation='h',
                      color='coefficient', color_continuous_scale='RdBu_r', color_continuous_midpoint=0)
    fig_coef.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_coef, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4: LTV FORECASTING
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("Lifetime Value Forecasting")
    
    # Cox PH results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cox PH Model")
        st.metric("Concordance Index", f"{ltv_results['cox_ph']['concordance_index']:.4f}")
        cox_features = pd.DataFrame(ltv_results['cox_ph']['top_features'])
        if 'covariate' in cox_features.columns:
            cox_features = cox_features.rename(columns={'covariate': 'Feature'})
        st.dataframe(cox_features, use_container_width=True)
    
    with col2:
        st.subheader("Kaplan-Meier Survival Curves")
        fig_km = go.Figure()
        colors = {'whale': '#ff0000', 'dolphin': '#ff9800', 'minnow': '#4caf50', 'free_rider': '#90a4ae'}
        for tier, data in ltv_results['kaplan_meier'].items():
            fig_km.add_trace(go.Scatter(
                x=data['timeline'], y=data['survival'],
                mode='lines', name=f"{tier} (median={data['median']:.0f}d)",
                line=dict(color=colors.get(tier, '#666'))
            ))
        fig_km.update_layout(height=350, margin=dict(t=20, b=20), 
                             xaxis_title="Days", yaxis_title="Survival Probability")
        st.plotly_chart(fig_km, use_container_width=True)
    
    # XGBoost LTV
    st.subheader("XGBoost LTV Prediction Performance")
    xgb_ltv = ltv_results['xgboost_ltv']
    col1, col2, col3 = st.columns(3)
    for i, (horizon, data) in enumerate(xgb_ltv.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"**{horizon.upper()}**")
            st.metric("RMSE", f"${data['rmse']:.2f}")
            st.metric("MAE", f"${data['mae']:.2f}")
            st.metric("R²", f"{data['r2']:.4f}")
    
    # Feature importance comparison
    st.subheader("Feature Importance Across LTV Horizons")
    fig_imp = go.Figure()
    for horizon, data in xgb_ltv.items():
        imp = data['feature_importance']
        fig_imp.add_trace(go.Bar(
            name=horizon.upper(),
            x=list(imp.keys())[:8], y=list(imp.values())[:8]
        ))
    fig_imp.update_layout(barmode='group', height=350, margin=dict(t=20, b=20))
    fig_imp.update_xaxes(tickangle=30)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # BG/NBD CLV
    st.subheader("BG/NBD + Gamma-Gamma CLV Model")
    clv_data = ltv_results['bgnbd_clv']
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers Modeled", f"{clv_data.get('n_customers', 'N/A'):,}")
    col2.metric("Avg CLV (90d)", f"${clv_data.get('avg_clv_90d', 0):.2f}")
    col3.metric("Median CLV (90d)", f"${clv_data.get('median_clv_90d', 0):.2f}")

# ══════════════════════════════════════════════════════════════
# TAB 5: A/B TESTING
# ══════════════════════════════════════════════════════════════
with tab5:
    st.header("A/B Testing & Experimentation")
    
    # Cookie Cats Results
    st.subheader("🐱 Cookie Cats: Gate Placement Experiment")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**D1 Retention**")
        d1 = ab_results['cookie_cats']['d1_retention']
        st.metric("Gate 30", f"{d1['p_a']:.3%}")
        st.metric("Gate 40", f"{d1['p_b']:.3%}")
        st.metric("Difference", f"{d1['diff']:+.4f}", delta=f"p={d1['p_value']:.4f}")
        sig = "✅ Significant" if d1['significant'] else "❌ Not Significant"
        st.markdown(f"**{sig}** at α=0.05")
    
    with col2:
        st.markdown("**D7 Retention**")
        d7 = ab_results['cookie_cats']['d7_retention']
        st.metric("Gate 30", f"{d7['p_a']:.3%}")
        st.metric("Gate 40", f"{d7['p_b']:.3%}")
        st.metric("Difference", f"{d7['diff']:+.4f}", delta=f"p={d7['p_value']:.4f}")
        sig = "✅ Significant" if d7['significant'] else "❌ Not Significant"
        st.markdown(f"**{sig}** at α=0.05")
    
    # Bootstrap CI
    st.subheader("Bootstrap Confidence Interval (D7 Retention Diff)")
    ci = ab_results['cookie_cats']['bootstrap_ci']
    fig_boot = go.Figure()
    fig_boot.add_trace(go.Scatter(x=[ci['lower'], ci['upper']], y=[1, 1], mode='lines',
                                   line=dict(width=8, color='#4ecdc4'), showlegend=False))
    fig_boot.add_trace(go.Scatter(x=[ci['mean_diff']], y=[1], mode='markers',
                                   marker=dict(size=15, color='#ff6b6b'), name='Mean Diff'))
    fig_boot.add_vline(x=0, line_dash="dash", line_color="red")
    fig_boot.update_layout(height=150, margin=dict(t=20, b=20), yaxis_visible=False,
                           xaxis_title="D7 Retention Difference (Gate 30 - Gate 40)")
    st.plotly_chart(fig_boot, use_container_width=True)
    
    # Bayesian Results
    st.subheader("Bayesian A/B Test (D7 Retention)")
    bay = ab_results['bayesian']
    col1, col2, col3 = st.columns(3)
    col1.metric("P(Gate 30 > Gate 40)", f"{bay['prob_a_better']:.1%}")
    col2.metric("Expected Loss (choose A)", f"{bay['expected_loss_a']:.5f}")
    col3.metric("95% Credible Interval", f"[{bay['credible_interval'][0]:.4f}, {bay['credible_interval'][1]:.4f}]")
    
    st.markdown("---")
    
    # CUPED Experiment
    st.subheader("💰 Pricing A/B Test with CUPED Variance Reduction")
    cuped = ab_results['cuped_experiment']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Standard Test**")
        st.metric("Revenue Diff", f"${cuped['raw']['diff']:.4f}")
        st.metric("Standard Error", f"{cuped['raw']['se']:.4f}")
        st.metric("p-value", f"{cuped['raw']['p_value']:.6f}")
    
    with col2:
        st.markdown("**CUPED-Adjusted**")
        st.metric("Revenue Diff", f"${cuped['cuped']['diff']:.4f}")
        st.metric("Standard Error", f"{cuped['cuped']['se']:.4f}")
        st.metric("p-value", f"{cuped['cuped']['p_value']:.6f}")
    
    with col3:
        st.markdown("**CUPED Benefit**")
        st.metric("Variance Reduction", f"{cuped['variance_reduction']:.1%}")
        st.metric("True Effect", f"{cuped['true_effect']:.0%}")
        se_reduction = 1 - cuped['cuped']['se'] / cuped['raw']['se']
        st.metric("SE Reduction", f"{se_reduction:.1%}")
    
    # Power Analysis
    st.subheader("📐 Power Analysis Calculator")
    power_df = pd.DataFrame(ab_results['power_analysis'])
    fig_power = px.line(power_df, x='mde', y='n_per_group', color='scenario', markers=True)
    fig_power.update_layout(height=350, margin=dict(t=20, b=20),
                            xaxis_title="Minimum Detectable Effect", yaxis_title="Sample Size per Group")
    st.plotly_chart(fig_power, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 6: CHURN & ALERTS
# ══════════════════════════════════════════════════════════════
with tab6:
    st.header("Churn Prediction & Live-Ops Alerts")
    
    # Model Performance
    st.subheader("Model Performance")
    models = churn_results['model_comparison']
    col1, col2, col3 = st.columns(3)
    col1.metric("XGBoost AUC", f"{models['xgboost']['auc']:.4f}")
    col2.metric("LightGBM AUC", f"{models['lightgbm']['auc']:.4f}")
    col3.metric("Best Model", models['best'])
    
    st.info("⚠️ Note: Perfect AUC is due to `days_since_last_active` being a direct proxy for the churn label. In production, this feature would be excluded or the churn definition would use a prediction window offset.")
    
    # SHAP Feature Importance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SHAP Feature Importance")
        shap_df = pd.DataFrame(churn_results['shap_importance'])
        fig_shap = px.bar(shap_df.head(12), y='feature', x='mean_abs_shap', orientation='h',
                          color='mean_abs_shap', color_continuous_scale='Reds')
        fig_shap.update_layout(height=400, margin=dict(t=20, b=20), yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_shap, use_container_width=True)
    
    with col2:
        st.subheader("Churn Risk Distribution")
        risk_data = churn_results['risk_distribution']
        risk_df = pd.DataFrame([{'risk': k, 'count': v} for k, v in risk_data.items()])
        risk_df = risk_df.sort_values('count', ascending=False)
        colors_map = {'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336', 'Critical': '#b71c1c'}
        fig_risk = px.pie(risk_df, values='count', names='risk',
                          color='risk', color_discrete_map=colors_map, hole=0.4)
        fig_risk.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix (XGBoost)")
    cm = np.array(churn_results['confusion_matrix'])
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=['Pred: Retained', 'Pred: Churned'], y=['Actual: Retained', 'Actual: Churned'],
        text=cm, texttemplate='%{text:,}', colorscale='Blues',
    ))
    fig_cm.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Live-Ops Recommendations
    st.subheader("🎯 Actionable Live-Ops Recommendations")
    for rec in churn_results['recommendations']:
        priority_colors = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
        icon = priority_colors.get(rec['priority'], '⚪')
        with st.expander(f"{icon} [{rec['priority']}] {rec['segment']} — {rec['count']:,} players"):
            st.markdown(f"**Action:** {rec['action']}")
            st.markdown(f"**Affected Players:** {rec['count']:,}")
    
    # Alert Thresholds
    st.subheader("⚡ Automated Alert Thresholds")
    alerts = churn_results['alert_thresholds']
    curr = churn_results['current_metrics']
    
    alert_status = []
    alert_status.append({
        'Metric': 'D1 Retention', 'Current': f"{curr['d1_retention']:.1%}",
        'Threshold': f">{alerts['d1_retention_min']:.0%}",
        'Status': '✅ OK' if curr['d1_retention'] >= alerts['d1_retention_min'] else '🚨 ALERT'
    })
    alert_status.append({
        'Metric': 'D7 Retention', 'Current': f"{curr['d7_retention']:.1%}",
        'Threshold': f">{alerts['d7_retention_min']:.0%}",
        'Status': '✅ OK' if curr['d7_retention'] >= alerts['d7_retention_min'] else '🚨 ALERT'
    })
    alert_status.append({
        'Metric': 'Conversion Rate', 'Current': f"{curr['conversion_rate']:.1%}",
        'Threshold': f">{alerts['conversion_rate_min']:.0%}",
        'Status': '✅ OK' if curr['conversion_rate'] >= alerts['conversion_rate_min'] else '🚨 ALERT'
    })
    alert_status.append({
        'Metric': 'Whales at Critical Risk', 'Current': str(curr['whales_at_critical_risk']),
        'Threshold': f"<{alerts['whale_churn_count']}",
        'Status': '✅ OK' if curr['whales_at_critical_risk'] < alerts['whale_churn_count'] else '🚨 ALERT'
    })
    
    st.dataframe(pd.DataFrame(alert_status), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🎮 Game Monetization & Live-Ops Analytics Platform | "
    "Built with Python, XGBoost, Cox Survival, BG/NBD, SHAP, Streamlit | "
    "310K+ Players Analyzed"
    "</div>",
    unsafe_allow_html=True
)
