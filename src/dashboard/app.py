"""
Game Monetization & Live-Ops Analytics Dashboard
==================================================
Works entirely from pre-computed JSON result files + raw CSVs.
No dependency on large processed CSVs (segmented_players.csv, master_features.csv).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, os, pathlib

st.set_page_config(page_title="Game Monetization & Live-Ops Analytics", page_icon="🎮", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>div[data-testid="stMetric"]{background-color:#0e1117;border:1px solid #262730;padding:15px;border-radius:10px;}.stTabs [data-baseweb="tab-list"]{gap:8px;}.stTabs [data-baseweb="tab"]{padding:10px 20px;border-radius:8px 8px 0 0;font-weight:600;}</style>""", unsafe_allow_html=True)

BASE = pathlib.Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
RAW = BASE / "data" / "raw"

@st.cache_data
def load_json(name):
    with open(PROCESSED / name) as f: return json.load(f)

@st.cache_data
def load_players():
    return pd.read_csv(RAW / "players.csv")

@st.cache_data
def load_txns():
    return pd.read_csv(RAW / "transactions.csv", parse_dates=['transaction_date'])

seg_results = load_json("segmentation_results.json")
funnel_results = load_json("funnel_results.json")
ltv_results = load_json("ltv_results.json")
ab_results = load_json("ab_testing_results.json")
churn_results = load_json("churn_results.json")
players = load_players()
txns = load_txns()

st.sidebar.title("🎮 Game Analytics")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Players:** {len(players):,}")
st.sidebar.markdown(f"**Payers:** {players['is_payer'].sum():,} ({players['is_payer'].mean():.1%})")
st.sidebar.markdown(f"**Total Revenue:** ${players['total_spend_usd'].sum():,.0f}")
st.sidebar.markdown(f"**Transactions:** {len(txns):,}")
st.sidebar.markdown("---")
platform_filter = st.sidebar.multiselect("Platform", players['platform'].unique(), default=list(players['platform'].unique()))
filtered = players[players['platform'].isin(platform_filter)]

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["📊 Executive Summary","👥 Player Segments","🔄 Conversion Funnels","💰 LTV Forecasting","🧪 A/B Testing","🚨 Churn & Alerts"])

with tab1:
    st.header("Executive Summary")
    m = churn_results.get('current_metrics',{})
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("DAU",f"{m.get('dau_estimate',0):,}"); c2.metric("MAU",f"{m.get('mau_estimate',0):,}")
    c3.metric("DAU/MAU",f"{m.get('dau_estimate',1)/max(m.get('mau_estimate',1),1):.1%}")
    c4.metric("Conversion",f"{m.get('conversion_rate',0):.1%}"); c5.metric("Whale %",f"{m.get('whale_pct',0):.2f}%")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("D1 Retention",f"{m.get('d1_retention',0):.1%}"); c2.metric("D7 Retention",f"{m.get('d7_retention',0):.1%}")
    c3.metric("D30 Retention",f"{m.get('d30_retention',0):.1%}"); c4.metric("ARPU",f"${m.get('arpu',0):.2f}"); c5.metric("ARPPU",f"${m.get('arppu',0):.2f}")
    st.markdown("---")
    co1,co2 = st.columns(2)
    with co1:
        st.subheader("Revenue by Player Tier")
        td = pd.DataFrame(seg_results['tier_revenue'])
        fig = px.pie(td[td['total_rev']>0],values='total_rev',names='player_tier',color_discrete_sequence=px.colors.sequential.Plasma_r,hole=0.4)
        fig.update_layout(height=350,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    with co2:
        st.subheader("Revenue Over Time")
        rw = txns.groupby(txns['transaction_date'].dt.to_period('W').astype(str))['price_usd'].sum().reset_index(); rw.columns=['week','revenue']
        fig = px.bar(rw,x='week',y='revenue',color_discrete_sequence=['#00d2ff'])
        fig.update_layout(height=350,margin=dict(t=20,b=20)); fig.update_xaxes(tickangle=45,nticks=20); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Cohort Retention Heatmap")
    filtered['ic'] = pd.to_datetime(filtered['install_date']).dt.to_period('M').astype(str)
    cr = filtered.groupby('ic').agg(n=('player_id','count'),d1=('d1_retained','mean'),d7=('d7_retained','mean'),d30=('d30_retained','mean')).reset_index()
    cr = cr[cr['n']>=100]; hd = cr[['ic','d1','d7','d30']].set_index('ic').T; hd.index=['D1','D7','D30']
    fig = go.Figure(data=go.Heatmap(z=hd.values*100,x=hd.columns,y=hd.index,colorscale='YlOrRd_r',text=np.round(hd.values*100,1),texttemplate='%{text:.1f}%',textfont={"size":11},colorbar=dict(title="Retention %")))
    fig.update_layout(height=250,margin=dict(t=10,b=10)); st.plotly_chart(fig,use_container_width=True)

with tab2:
    st.header("Player Segmentation & Whale Detection")
    co1,co2 = st.columns(2)
    with co1:
        st.subheader("K-Means Silhouette Analysis")
        sd = pd.DataFrame([{'k':int(k),'silhouette':v} for k,v in seg_results['silhouette_scores'].items()])
        fig = px.line(sd,x='k',y='silhouette',markers=True,color_discrete_sequence=['#00d2ff']); fig.update_layout(height=300,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    with co2:
        st.subheader("Revenue Concentration (Pareto)")
        td = pd.DataFrame(seg_results['tier_revenue']); td = td[td['total_rev']>0].sort_values('total_rev',ascending=False)
        fig = go.Figure(); fig.add_trace(go.Bar(x=td['player_tier'],y=td['pct_revenue'],name='% Revenue',marker_color='#ff6b6b'))
        fig.add_trace(go.Bar(x=td['player_tier'],y=td['pct_players'],name='% Players',marker_color='#4ecdc4'))
        fig.update_layout(barmode='group',height=300,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    st.subheader("RFM Segment Distribution (Payers)")
    rd = pd.DataFrame([{'segment':k,'count':v} for k,v in seg_results['rfm_distribution'].items()]).sort_values('count',ascending=True)
    fig = px.bar(rd,y='segment',x='count',orientation='h',color_discrete_sequence=['#7c4dff']); fig.update_layout(height=350,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    st.subheader("K-Means Cluster Profiles"); st.dataframe(pd.DataFrame(seg_results['cluster_profiles']),use_container_width=True)

with tab3:
    st.header("F2P Conversion Funnel Analysis")
    fd = pd.DataFrame(funnel_results['funnel'])
    fig = go.Figure(go.Funnel(y=fd['stage'],x=fd['count'],textposition="inside",textinfo="value+percent initial",marker=dict(color=['#667eea','#764ba2','#f093fb','#f5576c','#fda085','#f6d365','#96e6a1','#72c6ef'])))
    fig.update_layout(height=500,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    co1,co2 = st.columns(2)
    with co1:
        st.subheader("Drop-off Rates by Stage"); dd = fd[fd['dropoff_rate']>0]
        fig = px.bar(dd,x='stage',y='dropoff_rate',color='dropoff_rate',color_continuous_scale='Reds',text=dd['dropoff_rate'].apply(lambda x:f'{x:.1f}%'))
        fig.update_layout(height=350,margin=dict(t=20,b=20)); fig.update_xaxes(tickangle=30); st.plotly_chart(fig,use_container_width=True)
    with co2:
        st.subheader("Time-to-First-Purchase"); ttc = funnel_results['time_to_convert']['distribution']
        td = pd.DataFrame([{'bucket':k,'count':v} for k,v in ttc.items()])
        fig = px.bar(td,x='bucket',y='count',color_discrete_sequence=['#4ecdc4'],text='count'); fig.update_layout(height=350,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Conversion Rate by Country")
    cc = pd.DataFrame([{'country':k,'rate':v} for k,v in funnel_results['country_conversion'].items()]).sort_values('rate',ascending=False)
    fig = px.bar(cc,x='country',y='rate',color='rate',color_continuous_scale='Viridis',text=cc['rate'].apply(lambda x:f'{x:.2%}'))
    fig.update_layout(height=300,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Conversion Drivers (Logistic Regression)")
    cf = pd.DataFrame(funnel_results['logistic_regression']['coefficients']).sort_values('coefficient')
    fig = px.bar(cf,y='feature',x='coefficient',orientation='h',color='coefficient',color_continuous_scale='RdBu_r',color_continuous_midpoint=0)
    fig.update_layout(height=400,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)

with tab4:
    st.header("Lifetime Value Forecasting")
    co1,co2 = st.columns(2)
    with co1:
        st.subheader("Cox PH Model"); st.metric("Concordance Index",f"{ltv_results['cox_ph']['concordance_index']:.4f}")
        st.dataframe(pd.DataFrame(ltv_results['cox_ph']['top_features']),use_container_width=True)
    with co2:
        st.subheader("Kaplan-Meier Survival Curves"); fig = go.Figure()
        cols = {'whale':'#ff0000','dolphin':'#ff9800','minnow':'#4caf50','free_rider':'#90a4ae'}
        for t,d in ltv_results['kaplan_meier'].items():
            fig.add_trace(go.Scatter(x=d['timeline'],y=d['survival'],mode='lines',name=f"{t} ({d['median']:.0f}d)",line=dict(color=cols.get(t,'#666'))))
        fig.update_layout(height=350,margin=dict(t=20,b=20),xaxis_title="Days",yaxis_title="Survival Prob."); st.plotly_chart(fig,use_container_width=True)
    st.subheader("XGBoost LTV Prediction Performance"); xl = ltv_results['xgboost_ltv']
    co1,co2,co3 = st.columns(3)
    for i,(h,d) in enumerate(xl.items()):
        with [co1,co2,co3][i]: st.markdown(f"**{h.upper()}**"); st.metric("RMSE",f"${d['rmse']:.2f}"); st.metric("MAE",f"${d['mae']:.2f}"); st.metric("R²",f"{d['r2']:.4f}")
    st.subheader("Feature Importance Across Horizons"); fig = go.Figure()
    for h,d in xl.items():
        imp = d['feature_importance']; fig.add_trace(go.Bar(name=h.upper(),x=list(imp.keys())[:8],y=list(imp.values())[:8]))
    fig.update_layout(barmode='group',height=350,margin=dict(t=20,b=20)); fig.update_xaxes(tickangle=30); st.plotly_chart(fig,use_container_width=True)
    st.subheader("BG/NBD + Gamma-Gamma CLV"); cv = ltv_results['bgnbd_clv']
    co1,co2,co3 = st.columns(3); co1.metric("Customers",f"{cv.get('n_customers','N/A'):,}"); co2.metric("Avg CLV (90d)",f"${cv.get('avg_clv_90d',0):.2f}"); co3.metric("Median CLV",f"${cv.get('median_clv_90d',0):.2f}")

with tab5:
    st.header("A/B Testing & Experimentation")
    st.subheader("Cookie Cats: Gate Placement Experiment")
    co1,co2 = st.columns(2)
    with co1:
        st.markdown("**D1 Retention**"); d1 = ab_results['cookie_cats']['d1_retention']
        st.metric("Gate 30",f"{d1['p_a']:.3%}"); st.metric("Gate 40",f"{d1['p_b']:.3%}"); st.metric("Diff",f"{d1['diff']:+.4f}",delta=f"p={d1['p_value']:.4f}")
        st.markdown(f"**{'✅ Significant' if d1['significant'] else '❌ Not Significant'}**")
    with co2:
        st.markdown("**D7 Retention**"); d7 = ab_results['cookie_cats']['d7_retention']
        st.metric("Gate 30",f"{d7['p_a']:.3%}"); st.metric("Gate 40",f"{d7['p_b']:.3%}"); st.metric("Diff",f"{d7['diff']:+.4f}",delta=f"p={d7['p_value']:.4f}")
        st.markdown(f"**{'✅ Significant' if d7['significant'] else '❌ Not Significant'}**")
    st.subheader("Bootstrap 95% CI (D7 Diff)"); ci = ab_results['cookie_cats']['bootstrap_ci']
    fig = go.Figure(); fig.add_trace(go.Scatter(x=[ci['lower'],ci['upper']],y=[1,1],mode='lines',line=dict(width=8,color='#4ecdc4'),showlegend=False))
    fig.add_trace(go.Scatter(x=[ci['mean_diff']],y=[1],mode='markers',marker=dict(size=15,color='#ff6b6b'),name='Mean'))
    fig.add_vline(x=0,line_dash="dash",line_color="red"); fig.update_layout(height=150,margin=dict(t=20,b=20),yaxis_visible=False); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Bayesian A/B Test (D7)"); bay = ab_results['bayesian']
    co1,co2,co3 = st.columns(3)
    co1.metric("P(Gate 30 > Gate 40)",f"{bay['prob_a_better']:.1%}"); co2.metric("Expected Loss (A)",f"{bay['expected_loss_a']:.5f}")
    co3.metric("95% Credible Interval",f"[{bay['credible_interval'][0]:.4f}, {bay['credible_interval'][1]:.4f}]")
    st.markdown("---"); st.subheader("Pricing A/B Test with CUPED"); cp = ab_results['cuped_experiment']
    co1,co2,co3 = st.columns(3)
    with co1: st.markdown("**Standard**"); st.metric("Diff",f"${cp['raw']['diff']:.4f}"); st.metric("SE",f"{cp['raw']['se']:.4f}"); st.metric("p",f"{cp['raw']['p_value']:.2e}")
    with co2: st.markdown("**CUPED**"); st.metric("Diff",f"${cp['cuped']['diff']:.4f}"); st.metric("SE",f"{cp['cuped']['se']:.4f}"); st.metric("p",f"{cp['cuped']['p_value']:.2e}")
    with co3: st.markdown("**Benefit**"); st.metric("Var Reduction",f"{cp['variance_reduction']:.1%}"); st.metric("True Effect",f"{cp['true_effect']:.0%}"); st.metric("SE Reduction",f"{1-cp['cuped']['se']/cp['raw']['se']:.1%}")

with tab6:
    st.header("Churn Prediction & Live-Ops Alerts")
    md = churn_results['model_comparison']
    co1,co2,co3 = st.columns(3); co1.metric("XGBoost AUC",f"{md['xgboost']['auc']:.4f}"); co2.metric("LightGBM AUC",f"{md['lightgbm']['auc']:.4f}"); co3.metric("Best Model",md['best'])
    st.info("⚠️ Perfect AUC due to `days_since_last_active` feature leakage. In production, use temporal holdout with prediction window offset.")
    co1,co2 = st.columns(2)
    with co1:
        st.subheader("SHAP Feature Importance")
        sd = pd.DataFrame(churn_results['shap_importance']); fig = px.bar(sd.head(12),y='feature',x='mean_abs_shap',orientation='h',color='mean_abs_shap',color_continuous_scale='Reds')
        fig.update_layout(height=400,margin=dict(t=20,b=20),yaxis=dict(autorange='reversed')); st.plotly_chart(fig,use_container_width=True)
    with co2:
        st.subheader("Churn Risk Distribution")
        rd = pd.DataFrame([{'risk':k,'count':v} for k,v in churn_results['risk_distribution'].items()])
        fig = px.pie(rd,values='count',names='risk',color='risk',color_discrete_map={'Low':'#4caf50','Medium':'#ff9800','High':'#f44336','Critical':'#b71c1c'},hole=0.4)
        fig.update_layout(height=400,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    st.subheader("Confusion Matrix")
    cm = np.array(churn_results['confusion_matrix'])
    fig = go.Figure(data=go.Heatmap(z=cm,x=['Pred: Retained','Pred: Churned'],y=['Actual: Retained','Actual: Churned'],text=cm,texttemplate='%{text:,}',colorscale='Blues'))
    fig.update_layout(height=300,margin=dict(t=20,b=20)); st.plotly_chart(fig,use_container_width=True)
    st.subheader("🎯 Live-Ops Recommendations")
    for r in churn_results['recommendations']:
        ic = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢'}.get(r['priority'],'⚪')
        with st.expander(f"{ic} [{r['priority']}] {r['segment']} — {r['count']:,} players"): st.markdown(f"**Action:** {r['action']}")
    st.subheader("⚡ Alert Thresholds"); al = churn_results['alert_thresholds']; cu = churn_results['current_metrics']
    st.dataframe(pd.DataFrame([
        {'Metric':'D1 Retention','Current':f"{cu['d1_retention']:.1%}",'Threshold':f">{al['d1_retention_min']:.0%}",'Status':'✅ OK' if cu['d1_retention']>=al['d1_retention_min'] else '🚨 ALERT'},
        {'Metric':'D7 Retention','Current':f"{cu['d7_retention']:.1%}",'Threshold':f">{al['d7_retention_min']:.0%}",'Status':'✅ OK' if cu['d7_retention']>=al['d7_retention_min'] else '🚨 ALERT'},
        {'Metric':'Conversion','Current':f"{cu['conversion_rate']:.1%}",'Threshold':f">{al['conversion_rate_min']:.0%}",'Status':'✅ OK' if cu['conversion_rate']>=al['conversion_rate_min'] else '🚨 ALERT'},
        {'Metric':'Whales at Risk','Current':str(cu['whales_at_critical_risk']),'Threshold':f"<{al['whale_churn_count']}",'Status':'✅ OK' if cu['whales_at_critical_risk']<al['whale_churn_count'] else '🚨 ALERT'},
    ]),use_container_width=True,hide_index=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>🎮 Game Monetization & Live-Ops Analytics | Python, XGBoost, Cox Survival, BG/NBD, SHAP, Streamlit | 310K+ Players</div>",unsafe_allow_html=True)
