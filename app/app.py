import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="CLV Predictor",
    page_icon="💰",
    layout="wide"
)

# =========================
# Load Artifacts (cached)
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    num_scaler = joblib.load("num_scaler.pkl")
    feature_cols = joblib.load("model_feature_columns.pkl")
    num_cols = joblib.load("num_cols.pkl")
    return model, num_scaler, feature_cols, num_cols

model, num_scaler, model_feature_columns, num_cols = load_artifacts()

# =========================
# Helpers
# =========================
def inverse_log10(y_log: float) -> float:
    return float(10 ** y_log)

def build_encoded_row(
    recency, unique_products, total_items,
    avg_unitprice, max_unitprice,
    country, season
):
    # Start with all zeros in the exact schema (45 columns)
    row = pd.DataFrame([[0] * len(model_feature_columns)], columns=model_feature_columns)

    # Numeric fields
    row.loc[0, "Recency"] = recency
    row.loc[0, "unique_products"] = unique_products
    row.loc[0, "total_items"] = total_items
    row.loc[0, "avg_unitprice"] = avg_unitprice
    row.loc[0, "max_unitprice"] = max_unitprice

    # One-hot country/season
    country_col = f"most_common_country_{country}"
    season_col = f"most_common_season_{season}"

    if country_col in row.columns:
        row.loc[0, country_col] = 1
    if season_col in row.columns:
        row.loc[0, season_col] = 1

    # Scale numeric only (using numeric-only scaler)
    row[num_cols] = num_scaler.transform(row[num_cols])

    return row

def clv_tier(clv_value: float) -> str:
    if clv_value >= 100000:
        return "High-Value"
    elif clv_value >= 20000:
        return "Medium-Value"
    else:
        return "Low-Value"

def safe_percent(x):
    try:
        return float(x) * 100
    except:
        return None

# =========================
# Sidebar
# =========================
st.sidebar.title("💼 CLV Predictor")
st.sidebar.caption("XGBoost • log10(CLV) target • One-hot schema (45 features)")

with st.sidebar.expander("Model Artifacts", expanded=False):
    st.write(f"Features: **{len(model_feature_columns)}**")
    st.write(f"Scaled numeric: **{len(num_cols)}**")
    st.write(f"Model type: **{type(model).__name__}**")

st.sidebar.divider()

st.sidebar.subheader("Quick Tips")
st.sidebar.write("- Use **realistic values** for numeric fields.")
st.sidebar.write("- Choose a **country/season** seen in training.")
st.sidebar.write("- Output is inverse of **log10(CLV)**.")

# =========================
# Header
# =========================
st.title("💰 Customer Lifetime Value (CLV) Predictor")
st.markdown(
    "Predict CLV using a trained **XGBoost regression model**. "
    "The model was trained on **log10(CLV)** to reduce skewness."
)

tabs = st.tabs(["📌 Predict", "📊 Insights", "ℹ️ About"])

# =========================
# Tab 1: Prediction
# =========================
with tabs[0]:
    st.subheader("Customer Inputs")

    # country dropdown from feature schema
    country_options = sorted([
        c.replace("most_common_country_", "")
        for c in model_feature_columns
        if c.startswith("most_common_country_")
    ])
    season_options = ["winter", "spring", "summer", "autumn"]

    c1, c2, c3 = st.columns(3)

    with c1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
        unique_products = st.number_input("Unique Products Purchased", min_value=0, value=10, step=1)

    with c2:
        total_items = st.number_input("Total Items Purchased", min_value=0, value=200, step=10)
        avg_unitprice = st.number_input("Average Unit Price", min_value=0.0, value=3.5, step=0.1)

    with c3:
        max_unitprice = st.number_input("Maximum Unit Price", min_value=0.0, value=15.0, step=0.5)
        season = st.selectbox("Most Common Season", season_options, index=0)

    country = st.selectbox("Most Common Country", country_options,
                           index=country_options.index("United Kingdom") if "United Kingdom" in country_options else 0)

    st.divider()

    # Session history init
    if "history" not in st.session_state:
        st.session_state.history = []

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        predict_now = st.button("🚀 Predict CLV", use_container_width=True)

    with col_btn2:
        clear_hist = st.button("🧹 Clear History", use_container_width=True)

    if clear_hist:
        st.session_state.history = []
        st.success("History cleared.")

    if predict_now:
        X_ready = build_encoded_row(
            recency, unique_products, total_items,
            avg_unitprice, max_unitprice,
            country, season
        )

        y_log = float(model.predict(X_ready)[0])
        y_clv = inverse_log10(y_log)
        tier = clv_tier(y_clv)

        st.success("Prediction completed successfully ✅")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted log10(CLV)", f"{y_log:.4f}")
        m2.metric("Predicted CLV", f"${y_clv:,.2f}")
        m3.metric("Customer Tier", tier)

        # Add to history
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Recency": recency,
            "unique_products": unique_products,
            "total_items": total_items,
            "avg_unitprice": avg_unitprice,
            "max_unitprice": max_unitprice,
            "country": country,
            "season": season,
            "log10_clv_pred": y_log,
            "clv_pred": y_clv,
            "tier": tier
        })

        # Business note
        st.subheader("🧠 Business Interpretation")
        if tier == "High-Value":
            st.info("High-Value customer: prioritize retention, VIP offers, and personalized recommendations.")
        elif tier == "Medium-Value":
            st.info("Medium-Value customer: upsell/cross-sell bundles and encourage repeat purchases.")
        else:
            st.info("Low-Value customer: focus on activation campaigns and converting to repeat buyer.")

    # History table + download
    if len(st.session_state.history) > 0:
        st.divider()
        st.subheader("🗂 Prediction History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv,
            file_name="clv_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

# =========================
# Tab 2: Insights
# =========================
with tabs[1]:
    st.subheader("Model Insights")

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
        if len(importances) == len(model_feature_columns):
            fi = pd.DataFrame({
                "feature": model_feature_columns,
                "importance": importances
            }).sort_values("importance", ascending=False).head(15)

            st.write("Top 15 Feature Importances (XGBoost):")
            st.dataframe(fi, use_container_width=True)

            # Plot
            plt.figure()
            plt.barh(fi["feature"][::-1], fi["importance"][::-1])
            plt.title("Top Feature Importances")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.warning("Feature importance shape does not match feature columns.")
    else:
        st.warning("This model does not expose feature_importances_.")

    st.divider()
    st.subheader("Deployment Notes")
    st.write(
        "- Inputs are converted into a **fixed 45-column one-hot schema**.\n"
        "- Numeric columns are scaled using a **numeric-only StandardScaler**.\n"
        "- Output is inverse of **log10(CLV)**: `CLV = 10 ** prediction`."
    )

# =========================
# Tab 3: About
# =========================
with tabs[2]:
    st.subheader("About This App")
    st.write(
        "This Streamlit application is part of a customer analytics project for:\n"
        "- Customer segmentation\n"
        "- CLV prediction\n"
        "- Unsupervised + supervised + deep learning experiments\n\n"
        "**Deployment goal:** provide a lightweight interface to run the trained model and interpret results."
    )

    st.divider()
    st.subheader("Artifacts Used")
    st.code(
        "xgb_model.pkl\n"
        "num_scaler.pkl\n"
        "model_feature_columns.pkl\n"
        "num_cols.pkl"
    )
