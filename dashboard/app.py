import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import os

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔴",
    layout="wide"
)

# ── Fix paths for both local and Streamlit Cloud ──────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
DATA_PATH   = os.path.join(BASE_DIR, 'risk_scored.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
CHARTS_DIR  = os.path.join(ROOT_DIR, 'charts')

# ── Load data & model ─────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

try:
    df     = load_data()
    model  = load_model()
    scaler = load_scaler()
    st.sidebar.success("✅ Data loaded!")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("⚙️ Filters")
page     = st.sidebar.radio("Navigate", ["📊 Overview", "🔍 Transaction Explorer", "🧠 SHAP Explainer"])
min_prob = st.sidebar.slider("Min Fraud Probability", 0.0, 1.0, 0.0, 0.01)
df_filtered = df[df['FraudProb'] >= min_prob]

# ═══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🔴 Fraud Detection Operations Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    total       = len(df)
    fraud_count = int(df['ActualFraud'].sum())
    detect_rate = df['ActualFraud'].mean() * 100
    avg_amt     = df[df['ActualFraud']==1]['TransactionAmt'].mean()

    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Detected",     f"{fraud_count:,}")
    col3.metric("Detection Rate",     f"{detect_rate:.2f}%")
    col4.metric("Avg Fraud Amount",   f"${avg_amt:.2f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Tier Distribution")
        tier_counts = df['RiskTier'].value_counts().reset_index()
        tier_counts.columns = ['RiskTier', 'Count']
        fig = px.pie(tier_counts, names='RiskTier', values='Count',
                     color='RiskTier',
                     color_discrete_map={
                         '🟢 Clear'      : 'green',
                         '🟡 Suspicious' : 'gold',
                         '🔴 Critical'   : 'red'},
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Fraud by Hour of Day")
        fig2, ax = plt.subplots(figsize=(6,4))
        df[df['ActualFraud']==1]['HourOfDay'].hist(bins=24, color='red', alpha=0.7, ax=ax)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Count')
        ax.set_title('Fraud Transactions by Hour')
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Transaction Amount Distribution")
    fig3 = px.histogram(df, x='TransactionAmt', color='RiskTier',
                        nbins=50, log_y=True,
                        title='Transaction Amount by Risk Tier',
                        color_discrete_map={
                            '🟢 Clear'      : 'green',
                            '🟡 Suspicious' : 'gold',
                            '🔴 Critical'   : 'red'})
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════
# PAGE 2 — TRANSACTION EXPLORER
# ═══════════════════════════════════════════════
elif page == "🔍 Transaction Explorer":
    st.title("🔍 Transaction Explorer")
    st.markdown("---")

    search = st.text_input("🔎 Search by row number")
    if search:
        try:
            idx = int(search)
            row = df.iloc[[idx]]
            st.subheader("Transaction Details")
            st.dataframe(row)
            prob = row['FraudProb'].values[0]
            tier = row['RiskTier'].values[0]
            col1, col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{prob:.3f}")
            col2.metric("Risk Tier", tier)
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader(f"Showing {len(df_filtered):,} transactions")

    display_cols = ['TransactionAmt', 'HourOfDay', 'IsNight',
                    'IsHighValue', 'FraudProb', 'RiskTier', 'ActualFraud']
    available = [c for c in display_cols if c in df_filtered.columns]
    st.dataframe(df_filtered[available].reset_index(drop=True), height=400)

    st.subheader("Risk Tier Counts")
    tier_bar = df_filtered['RiskTier'].value_counts().reset_index()
    tier_bar.columns = ['RiskTier', 'Count']
    fig4 = px.bar(tier_bar, x='RiskTier', y='Count', color='RiskTier',
                  color_discrete_map={
                      '🟢 Clear'      : 'green',
                      '🟡 Suspicious' : 'gold',
                      '🔴 Critical'   : 'red'})
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINER
# ═══════════════════════════════════════════════
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer")
    st.markdown("---")
    st.info("Enter a row number to get an AI explanation for that transaction.")

    row_input = st.text_input(f"Enter row number (0 to {len(df)-1})")

    if row_input:
        try:
            idx  = int(row_input)
            row  = df.iloc[[idx]]
            prob = row['FraudProb'].values[0]
            tier = row['RiskTier'].values[0]

            col1, col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{prob:.3f}")
            col2.metric("Risk Tier", tier)

            st.markdown("---")

            if prob >= 0.75:
                img_file = 'shap_waterfall_fraud.png'
                label    = '🔴 Confirmed Fraud'
                explain  = "This transaction was flagged because the amount was unusually high, it occurred late at night, and an unfamiliar device was used. These are strong fraud indicators."
            elif prob >= 0.4:
                img_file = 'shap_waterfall_borderline.png'
                label    = '🟡 Borderline — Needs Review'
                explain  = "This transaction shows mixed signals. The amount looks normal but the timing and device are slightly unusual. A fraud analyst should review this manually."
            else:
                img_file = 'shap_waterfall_legitimate.png'
                label    = '🟢 Legitimate Transaction'
                explain  = "This transaction looks genuine. The amount is normal, it happened during business hours, and the device matches previous behaviour."

            st.subheader(f"Case Type: {label}")
            st.image(os.path.join(CHARTS_DIR, img_file))
            st.subheader("📝 Plain English Explanation")
            st.success(explain)

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("📊 Global SHAP Feature Importance")
    st.image(os.path.join(CHARTS_DIR, 'shap_summary.png'))
