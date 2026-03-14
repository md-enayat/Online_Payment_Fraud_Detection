"""
================================================================================
 FRAUD DETECTION PIPELINE — STREAMLIT APP
================================================================================
 File        : app.py
 Project     : Online Payment Fraud Detection
 Description : Professional Streamlit web application for live fraud
               detection. User inputs transaction details, model predicts
               fraud probability with confidence level and risk factors.

 Imports from: src/predict.py, src/config.py
 Run with    : streamlit run app/app.py
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# ── Add project root to path ───────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Import from pipeline ───────────────────────────────────────────────────────
from src.predict import predict_single, load_model_and_preprocessor


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title = "Fraud Detection System",
    page_icon  = "🔍",
    layout     = "wide",
)


# ==============================================================================
# CUSTOM CSS — PROFESSIONAL STYLING
# ==============================================================================

st.markdown("""
<style>
    /* ── Main background ── */
    .main { background-color: #0e1117; }

    /* ── Header ── */
    .header-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #e94560;
    }
    .header-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* ── Section headers ── */
    .section-header {
        color: #e94560;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }

    /* ── Result cards ── */
    .fraud-card {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    .legit-card {
        background: linear-gradient(135deg, #00c851, #007e33);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .result-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e94560;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
    }

    /* ── Risk factor items ── */
    .risk-item-high {
        background: #2d1515;
        border-left: 4px solid #ff4444;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        color: #ff6b6b;
    }
    .risk-item-medium {
        background: #2d2515;
        border-left: 4px solid #ffaa00;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        color: #ffd166;
    }
    .risk-item-low {
        background: #152d15;
        border-left: 4px solid #00c851;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        color: #69f0ae;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# LOAD MODEL — CACHED SO IT LOADS ONLY ONCE
# ==============================================================================

@st.cache_resource
def load_model():
    """
    Load model, preprocessor and features once.
    Cached by Streamlit — not reloaded on every interaction.
    """
    return load_model_and_preprocessor()


# ==============================================================================
# HELPER — FRAUD PROBABILITY GAUGE
# ==============================================================================

def render_gauge(probability: float):
    """
    Render a Plotly gauge chart showing fraud probability.

    Args:
        probability (float): Fraud probability (0.0 - 1.0)
    """

    color = "#ff4444" if probability >= 0.5 else "#00c851"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = probability * 100,
        title = {"text": "Fraud Probability %", "font": {"color": "white"}},
        number= {"suffix": "%", "font": {"color": color, "size": 40}},
        gauge = {
            "axis"    : {"range": [0, 100], "tickcolor": "white"},
            "bar"     : {"color": color},
            "bgcolor" : "#1a1a2e",
            "steps"   : [
                {"range": [0,  40],  "color": "#152d15"},
                {"range": [40, 70],  "color": "#2d2515"},
                {"range": [70, 100], "color": "#2d1515"},
            ],
            "threshold": {
                "line" : {"color": "white", "width": 4},
                "value": 50,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        font_color   ="white",
        height       =300,
        margin       =dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# HELPER — RISK FACTORS ANALYSIS
# ==============================================================================

def render_risk_factors(input_data: dict):
    """
    Analyze and display key risk factors from the transaction.
    Highlights suspicious values that indicate potential fraud.

    Args:
        input_data (dict): Transaction input values
    """

    st.markdown(
        '<p class="section-header">⚠️ Risk Factor Analysis</p>',
        unsafe_allow_html=True
    )

    risk_factors = []

    # ── Check each risk factor ─────────────────────────────────────────────────

    # New account — high fraud risk
    if input_data["account_age_days"] < 30:
        risk_factors.append(("HIGH", "🔴 Very new account (< 30 days) — high fraud risk"))
    elif input_data["account_age_days"] < 90:
        risk_factors.append(("MEDIUM", "🟡 Relatively new account (< 90 days)"))
    else:
        risk_factors.append(("LOW", "🟢 Established account — low risk"))

    # High IP risk score
    if input_data["ip_risk_score"] > 0.7:
        risk_factors.append(("HIGH", "🔴 Very high IP risk score — suspicious origin"))
    elif input_data["ip_risk_score"] > 0.4:
        risk_factors.append(("MEDIUM", "🟡 Moderate IP risk score"))
    else:
        risk_factors.append(("LOW", "🟢 Low IP risk score"))

    # Transaction amount deviation
    if input_data["amount_deviation_from_user_mean"] > 5000:
        risk_factors.append(("HIGH", "🔴 Transaction amount far above user average"))
    elif input_data["amount_deviation_from_user_mean"] > 2000:
        risk_factors.append(("MEDIUM", "🟡 Transaction amount above user average"))
    else:
        risk_factors.append(("LOW", "🟢 Transaction amount within normal range"))

    # High velocity — many transactions in 1 hour
    if input_data["txn_count_1h"] > 5:
        risk_factors.append(("HIGH", "🔴 High transaction velocity (> 5 in 1 hour)"))
    elif input_data["txn_count_1h"] > 3:
        risk_factors.append(("MEDIUM", "🟡 Moderate transaction velocity"))
    else:
        risk_factors.append(("LOW", "🟢 Normal transaction velocity"))

    # Failed transactions
    if input_data["failed_txn_count_24h"] > 3:
        risk_factors.append(("HIGH", "🔴 Multiple failed transactions in 24h"))
    elif input_data["failed_txn_count_24h"] > 1:
        risk_factors.append(("MEDIUM", "🟡 Some failed transactions in 24h"))
    else:
        risk_factors.append(("LOW", "🟢 No recent failed transactions"))

    # International transaction
    if input_data["is_international"] == 1:
        risk_factors.append(("MEDIUM", "🟡 International transaction detected"))
    else:
        risk_factors.append(("LOW", "🟢 Domestic transaction"))

    # Geo distance
    if input_data["geo_distance_from_last_txn"] > 500:
        risk_factors.append(("HIGH", "🔴 Suspicious location jump (> 500km from last txn)"))
    elif input_data["geo_distance_from_last_txn"] > 100:
        risk_factors.append(("MEDIUM", "🟡 Moderate distance from last transaction"))
    else:
        risk_factors.append(("LOW", "🟢 Normal location pattern"))

    # Merchant risk
    if input_data["merchant_risk_score"] > 0.7:
        risk_factors.append(("HIGH", "🔴 High risk merchant"))
    elif input_data["merchant_risk_score"] > 0.4:
        risk_factors.append(("MEDIUM", "🟡 Moderate risk merchant"))
    else:
        risk_factors.append(("LOW", "🟢 Low risk merchant"))

    # ── Render risk factors ────────────────────────────────────────────────────
    for level, message in risk_factors:
        if level == "HIGH":
            st.markdown(
                f'<div class="risk-item-high">{message}</div>',
                unsafe_allow_html=True
            )
        elif level == "MEDIUM":
            st.markdown(
                f'<div class="risk-item-medium">{message}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="risk-item-low">{message}</div>',
                unsafe_allow_html=True
            )


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="header-container">
        <p class="header-title">🔍 Online Payment Fraud Detection</p>
        <p class="header-subtitle">
            AI-powered real-time fraud detection system
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    model, preprocessor, features = load_model()

    # ── Layout — two columns ───────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    # ==========================================================================
    # LEFT COLUMN — INPUT FORM
    # ==========================================================================

    with left_col:

        st.markdown(
            '<p class="section-header">📋 Transaction Details</p>',
            unsafe_allow_html=True
        )

        # ── Account Information ────────────────────────────────────────────────
        st.markdown("**Account Information**")
        col1, col2 = st.columns(2)

        with col1:
            account_age_days = st.number_input(
                "Account Age (days)",
                min_value=0, max_value=3650,
                value=365,
                help="Number of days since account was created"
            )
            credit_score_band = st.selectbox(
                "Credit Score Band",
                options=[1, 2, 3, 4, 5],
                index=2,
                help="1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent"
            )

        with col2:
            kyc_level = st.selectbox(
                "KYC Level",
                options=[1, 2, 3],
                index=1,
                help="1=Basic, 2=Standard, 3=Enhanced"
            )
            is_international = st.selectbox(
                "International Transaction",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Is this an international transaction?"
            )

        st.markdown("---")

        # ── Transaction Information ────────────────────────────────────────────
        st.markdown("**Transaction Information**")
        col3, col4 = st.columns(2)

        with col3:
            transaction_amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0, max_value=100000.0,
                value=500.0, step=10.0,
            )
            avg_monthly_spend = st.number_input(
                "Avg Monthly Spend ($)",
                min_value=0.0, max_value=100000.0,
                value=3000.0, step=100.0,
            )
            payment_channel = st.selectbox(
                "Payment Channel",
                options=["card", "upi", "wallet", "netbanking"],
            )

        with col4:
            device_type = st.selectbox(
                "Device Type",
                options=["mobile", "desktop", "tablet"],
            )
            txn_count_1h = st.number_input(
                "Transactions (last 1h)",
                min_value=0, max_value=50, value=1,
            )
            txn_count_24h = st.number_input(
                "Transactions (last 24h)",
                min_value=0, max_value=100, value=3,
            )

        st.markdown("---")

        # ── Risk Scores ────────────────────────────────────────────────────────
        st.markdown("**Risk Scores**")
        col5, col6 = st.columns(2)

        with col5:
            ip_risk_score = st.slider(
                "IP Risk Score",
                min_value=0.0, max_value=1.0,
                value=0.1, step=0.01,
            )
            merchant_risk_score = st.slider(
                "Merchant Risk Score",
                min_value=0.0, max_value=1.0,
                value=0.1, step=0.01,
            )
            post_auth_risk_score = st.slider(
                "Post Auth Risk Score",
                min_value=0.0, max_value=1.0,
                value=0.1, step=0.01,
            )

        with col6:
            failed_txn_count_24h = st.number_input(
                "Failed Transactions (24h)",
                min_value=0, max_value=20, value=0,
            )
            geo_distance_from_last_txn = st.number_input(
                "Geo Distance from Last Txn (km)",
                min_value=0.0, max_value=20000.0,
                value=5.0, step=1.0,
            )
            amount_deviation_from_user_mean = st.number_input(
                "Amount Deviation from User Mean ($)",
                min_value=0.0, max_value=100000.0,
                value=100.0, step=10.0,
            )

        st.markdown("---")

        # ── Predict Button ─────────────────────────────────────────────────────
        predict_btn = st.button(
            "🔍 Analyze Transaction",
            use_container_width=True,
            type="primary",
        )

    # ==========================================================================
    # RIGHT COLUMN — RESULTS
    # ==========================================================================

    with right_col:

        if predict_btn:

            # ── Build input dict ───────────────────────────────────────────────
            input_data = {
                "account_age_days"                : account_age_days,
                "credit_score_band"               : credit_score_band,
                "kyc_level"                       : kyc_level,
                "avg_monthly_spend"               : avg_monthly_spend,
                "merchant_risk_score"             : merchant_risk_score,
                "transaction_amount"              : transaction_amount,
                "payment_channel"                 : payment_channel,
                "device_type"                     : device_type,
                "is_international"                : is_international,
                "ip_risk_score"                   : ip_risk_score,
                "txn_count_1h"                    : txn_count_1h,
                "txn_count_24h"                   : txn_count_24h,
                "failed_txn_count_24h"            : failed_txn_count_24h,
                "geo_distance_from_last_txn"      : geo_distance_from_last_txn,
                "amount_deviation_from_user_mean" : amount_deviation_from_user_mean,
                "post_auth_risk_score"            : post_auth_risk_score,
            }

            # ── Run prediction ─────────────────────────────────────────────────
            with st.spinner("Analyzing transaction..."):
                result = predict_single(
                    input_data, model, preprocessor, features
                )

            # ── Result card ────────────────────────────────────────────────────
            if result["label"] == 1:
                st.markdown(f"""
                <div class="fraud-card">
                    <p class="result-title">🚨 FRAUD DETECTED</p>
                    <p class="result-subtitle">
                        This transaction has been flagged as fraudulent
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-card">
                    <p class="result-title">✅ LEGITIMATE</p>
                    <p class="result-subtitle">
                        This transaction appears to be legitimate
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metrics row ────────────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{result['probability']*100:.1f}%</p>
                    <p class="metric-label">Fraud Probability</p>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{result['confidence']}</p>
                    <p class="metric-label">Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                risk_score = "High" if result["probability"] >= 0.7 \
                             else "Medium" if result["probability"] >= 0.4 \
                             else "Low"
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{risk_score}</p>
                    <p class="metric-label">Risk Level</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauge chart ────────────────────────────────────────────────────
            render_gauge(result["probability"])

            # ── Transaction summary ────────────────────────────────────────────
            st.markdown(
                '<p class="section-header">📊 Transaction Summary</p>',
                unsafe_allow_html=True
            )

            summary_df = pd.DataFrame({
                "Feature" : [
                    "Transaction Amount",
                    "Payment Channel",
                    "Device Type",
                    "Account Age",
                    "IP Risk Score",
                    "Merchant Risk Score",
                    "International",
                    "Txn Count (1h)",
                    "Failed Txn (24h)",
                ],
                "Value": [
                    f"${transaction_amount:,.2f}",
                    payment_channel.upper(),
                    device_type.capitalize(),
                    f"{account_age_days} days",
                    f"{ip_risk_score:.2f}",
                    f"{merchant_risk_score:.2f}",
                    "Yes" if is_international else "No",
                    txn_count_1h,
                    failed_txn_count_24h,
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # ── Risk factors ───────────────────────────────────────────────────
            render_risk_factors(input_data)

        else:
            # ── Placeholder when no prediction yet ────────────────────────────
            st.markdown("""
            <div style="
                text-align: center;
                padding: 4rem 2rem;
                color: #4a5568;
                border: 2px dashed #2d3748;
                border-radius: 12px;
                margin-top: 2rem;
            ">
                <p style="font-size: 3rem;">🔍</p>
                <p style="font-size: 1.2rem; font-weight: 600;">
                    Fill in transaction details
                </p>
                <p style="font-size: 0.9rem;">
                    and click Analyze Transaction to get prediction
                </p>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# RUN APP
# ==============================================================================

if __name__ == "__main__":
    main()