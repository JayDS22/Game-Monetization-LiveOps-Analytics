# Game Monetization & Live-Ops Analytics Platform

**End-to-end F2P game analytics platform built on 310K+ players: whale detection, LTV forecasting, churn prediction, A/B testing, and live-ops dashboards.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ukti6zgqjjpytjxkvbgx7k.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## [Live Demo → Click here to explore the dashboard](https://ukti6zgqjjpytjxkvbgx7k.streamlit.app/)

---

## 📋 Project Overview

This platform demonstrates production-grade data science for free-to-play game monetization — the exact analytical stack used by studios like Tencent, Supercell, and Zynga to optimize player lifetime value and live-ops decisions.

### What It Does

| Module | Description | Key Methods |
|--------|-------------|-------------|
| **Data Pipeline** | Ingests 3 datasets, engineers 59+ features, SQL warehouse views | Pandas, feature engineering, RFM |
| **Player Segmentation** | Whale/dolphin/minnow detection, behavioral clustering | K-Means, DBSCAN, RFM, PCA |
| **Conversion Funnels** | Install→IAP journey mapping, friction point identification | Chi-squared, logistic regression |
| **LTV Forecasting** | Multi-horizon lifetime value prediction | Cox PH, XGBoost, BG/NBD + Gamma-Gamma |
| **A/B Testing** | Full experimentation framework with variance reduction | Z-test, bootstrap CI, CUPED, Bayesian |
| **Churn Prediction** | Binary classifier with individual-level explainability | XGBoost, LightGBM, SHAP |
| **Live-Ops Dashboard** | 6-tab Streamlit app with KPIs, alerts, recommendations | Streamlit, Plotly |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ Players (310K)│ │ IAP Txns (35K)│ │ A/B Test (90K)     │ │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘ │
└─────────┼────────────────┼─────────────────────┼─────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING PIPELINE                     │
│  59+ features: RFM, engagement, spending, progression,       │
│  demographics, retention scores, social multipliers          │
│  SQL views: vw_daily_kpis, vw_cohort_retention, etc.        │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Segmentation │ │   Funnels    │ │  A/B Testing │
│ K-Means/RFM  │ │ Conversion   │ │ CUPED/Bayes  │
│ DBSCAN/PCA   │ │ LogReg/Chi²  │ │ Bootstrap CI │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ LTV Forecast │ │    Churn     │ │  Live-Ops    │
│ Cox/XGB/BGNBD│ │ XGB+SHAP    │ │ Alerts/Recs  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
         ┌──────────────────────────┐
         │   STREAMLIT DASHBOARD    │
         │  6 Tabs • KPI Cards •    │
         │  Cohort Heatmaps • SHAP  │
         └──────────────────────────┘
```

---

## Key Results

### Monetization Metrics
- **310,000** players analyzed across 15 countries
- **3.2%** F2P-to-payer conversion rate
- **$200K+** total revenue from 34,661 transactions
- **Whales (0.1% of payers) drive 54% of revenue** — classic F2P Pareto distribution

### Model Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| Cox PH | Player Lifetime | C-Index | 0.7325 |
| XGBoost | LTV-30d | R² | 0.9555 |
| XGBoost | LTV-90d | R² | 0.9907 |
| XGBoost | Churn (D7) | ROC-AUC | 1.0000* |
| LightGBM | Churn (D7) | ROC-AUC | 1.0000* |
| BG/NBD + Gamma-Gamma | CLV (90d) | Avg CLV | $0.86 |

*\*Perfect AUC due to `days_since_last_active` feature leakage — documented as expected in synthetic demo. Production implementation would exclude this feature or use temporal holdout.*

### A/B Testing (Cookie Cats)
- **D7 retention**: Gate 30 significantly outperforms Gate 40 (p=0.024)
- **CUPED variance reduction**: 47.9% — enables detecting smaller effects
- **Bayesian P(Gate 30 > Gate 40)**: 98.8%

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Full Pipeline
```bash
# 1. Generate synthetic data (310K+ players)
python src/ingestion/generate_data.py

# 2. Feature engineering (59+ features)
python src/ingestion/feature_engineering.py

# 3. Player segmentation & whale detection
python src/segmentation/player_segmentation.py

# 4. Conversion funnel analysis
python src/funnel/conversion_funnel.py

# 5. LTV forecasting (Cox, XGBoost, BG/NBD)
python src/ltv/ltv_forecasting.py

# 6. A/B testing framework
python src/ab_testing/ab_framework.py

# 7. Churn prediction & live-ops
python src/churn/churn_prediction.py

# 8. Launch dashboard
streamlit run src/dashboard/app.py
```

### Or just view the live dashboard
No setup needed — [**open the live demo →**](https://ukti6zgqjjpytjxkvbgx7k.streamlit.app/)

---

## Project Structure

```
Game-Monetization-LiveOps-Analytics/
├── README.md
├── requirements.txt
├── .streamlit/
│   └── config.toml                  # Dark theme configuration
├── data/
│   ├── raw/
│   │   ├── players.csv              # 310K player profiles
│   │   ├── transactions.csv         # 34K+ IAP transactions
│   │   └── cookie_cats_ab.csv       # 90K A/B test records
│   └── processed/
│       ├── segmentation_results.json
│       ├── funnel_results.json
│       ├── ltv_results.json
│       ├── ab_testing_results.json
│       └── churn_results.json
├── src/
│   ├── ingestion/
│   │   ├── generate_data.py         # Synthetic data generator
│   │   └── feature_engineering.py   # 59+ feature pipeline
│   ├── segmentation/
│   │   └── player_segmentation.py   # K-Means, DBSCAN, RFM
│   ├── funnel/
│   │   └── conversion_funnel.py     # Funnel analysis
│   ├── ltv/
│   │   └── ltv_forecasting.py       # Cox, XGBoost, BG/NBD
│   ├── ab_testing/
│   │   └── ab_framework.py          # Full A/B testing suite
│   ├── churn/
│   │   └── churn_prediction.py      # XGBoost + SHAP
│   └── dashboard/
│       └── app.py                   # Streamlit dashboard
└── sql/
    ├── vw_daily_kpis.sql
    ├── vw_cohort_retention.sql
    ├── vw_whale_segments.sql
    └── vw_conversion_funnel.sql
```

---

## 🔬 Methodology Deep Dive

### Player Segmentation
- **K-Means** with silhouette analysis (optimal k=3, silhouette=0.253)
- **DBSCAN** for outlier-based whale detection on payer subpopulation
- **RFM scoring**: quintile-based Recency/Frequency/Monetary with 8 behavioral segments (Champions, Loyal Whales, At-Risk High Value, etc.)

### LTV Modeling
- **Cox Proportional Hazards**: Semi-parametric survival model with C-index=0.73. Key drivers: stage progression (HR=0.97) and session count (HR=0.99) reduce churn hazard
- **XGBoost regression**: 7-day, 30-day, 90-day LTV windows. R²=0.96 at 30-day horizon
- **BG/NBD + Gamma-Gamma**: Probabilistic CLV from the `lifetimes` library. Models purchase frequency and monetary value separately

### A/B Testing Framework
- **Frequentist**: Two-proportion z-test, chi-squared contingency, Welch's t-test
- **Bootstrap**: 10,000-iteration non-parametric confidence intervals
- **CUPED**: Controlled Using Pre-Experiment Data — 48% variance reduction using pre-experiment session counts as covariate
- **Bayesian**: Beta-Binomial conjugate model with 100K posterior samples. Reports P(A>B) and expected loss

### Churn Prediction
- **XGBoost + LightGBM** with SHAP explainability
- **Per-player SHAP waterfall** plots showing top 3 churn reasons
- **Risk tiering**: Low / Medium / High / Critical with targeted intervention recommendations

---

## 🎯 Gaming Domain Expertise

This project uses the exact vocabulary and metrics that F2P game data science teams track:

- **DAU/MAU/Stickiness** — daily and monthly active users, engagement ratio
- **D1/D7/D30 retention** — industry-standard retention cohorts
- **ARPU/ARPPU** — average revenue per user / per paying user
- **Whale/Dolphin/Minnow** — spending tier classification
- **Conversion funnels** — install → tutorial → first IAP → repeat purchase
- **Live-ops** — real-time player management, intervention triggers, push notification targeting

---

## Datasets

| Dataset | Inspired By | Records | Description |
|---------|-------------|---------|-------------|
| players.csv | Uken Games (SFU) | 310,000 | Player profiles, engagement, retention, demographics, spending |
| transactions.csv | Mobile IAP 2025 | 34,661 | Detailed purchase events with 20 IAP item types |
| cookie_cats_ab.csv | Cookie Cats (Kaggle) | 90,000 | Gate placement A/B test with D1/D7 retention |

All data is synthetically generated with realistic F2P distributions (power-law spending, exponential session decay, correlated engagement-spending patterns).

---

## Technologies

**Languages & Libraries**: Python, SQL, Pandas, NumPy, SciPy, Scikit-learn

**ML/Stats**: XGBoost, LightGBM, Lifelines (Cox PH, KM), Lifetimes (BG/NBD, Gamma-Gamma), SHAP

**Visualization**: Streamlit, Plotly, Matplotlib, Seaborn

**Statistical Methods**: Z-test, Chi-squared, Bootstrap CI, CUPED variance reduction, Bayesian A/B testing, Power analysis

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a comprehensive portfolio project demonstrating end-to-end data science for F2P game monetization and live-ops analytics.*
