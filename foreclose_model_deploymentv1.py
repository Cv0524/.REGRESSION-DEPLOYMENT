import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(page_title="Minimum Bid Price Predictor", layout="wide")

# ==========================
# Global CSS (polished)
# ==========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    .stApp { background: #F6F8FF; font-family: "Poppins","Inter","Segoe UI",Arial,sans-serif; }

    .title-banner {
        background: linear-gradient(90deg, #0F172A 0%, #1D4ED8 60%, #0EA5E9 100%);
        padding: 18px 18px; border-radius: 14px; color: white;
        font-size: 34px; font-weight: 800; margin-bottom: 12px;
        box-shadow: 0 10px 26px rgba(2, 6, 23, 0.22);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0F172A 0%, #1D4ED8 60%, #0EA5E9 100%) !important;
        border: 0 !important; color: #FFF !important; font-weight: 800 !important;
        border-radius: 12px !important; padding: 0.70rem 1rem !important;
        box-shadow: 0 12px 26px rgba(2, 6, 23, 0.22) !important;
        transition: transform .08s ease-in-out, box-shadow .08s ease-in-out;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(2, 6, 23, 0.28) !important;
        filter: saturate(1.05);
    }

    label[data-testid="stMetricLabel"] p { font-size: 0.95rem !important; color: rgba(15,23,42,.85) !important; }
    div[data-testid="stMetricValue"] p { font-size: 1.0rem !important; color: #0F172A !important; }
    </style>

    <div class="title-banner">🏠 Foreclosed Property – Minimum Bid Price Predictor</div>
    """,
    unsafe_allow_html=True,
)

CATEGORY_MAPS = {
    "REGION": {"METRO MANILA": 0, "NON-METRO MANILA": 1},
    "REMARKS": {"TCT UNDER THE BANK": 0, "FOR TITLE CONSOLIDATION": 1},
    "STATUS": {"UNOCCUPIED": 0, "OCCUPIED": 1},
}
DEFAULT_ENCODED_VALUES = {}

# Base directory = where this script lives (good for Streamlit Cloud)
BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts(base_dir: Path):
    feature_cols = joblib.load(base_dir / "feature_cols.pkl")
    lot_area_q75 = joblib.load(base_dir / "lot_area_q75.pkl")
    scaler = joblib.load(base_dir / "scaler.pkl")

    models = {
        "Linear Reg": joblib.load(base_dir / "linear_reg.pkl"),
        "Ridge (Tuned)": joblib.load(base_dir / "ridge_tuned.pkl"),
        "Lasso (Tuned)": joblib.load(base_dir / "lasso_tuned.pkl"),
        "Random Forest (Tuned)": joblib.load(base_dir / "rf_tuned.pkl"),
        "Gradient Boosting (Tuned)": joblib.load(base_dir / "gb_tuned.pkl"),
    }
    return feature_cols, lot_area_q75, scaler, models


def build_features_from_user(user_input: dict, feature_cols, lot_area_q75) -> pd.DataFrame:
    df = pd.DataFrame([user_input]).copy()

    for col in ["LOT AREA (sqm)", "FLOOR AREA (sqm)"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    lot_area_orig = np.expm1(df["LOT AREA (sqm)"])
    floor_orig = np.expm1(df["FLOOR AREA (sqm)"])

    df["TOTAL_AREA"] = df["LOT AREA (sqm)"] + df["FLOOR AREA (sqm)"]
    df["FLOOR_TO_LOT_RATIO"] = np.log1p(floor_orig / (lot_area_orig + 1))
    df["LOG_LOT_x_FLOOR"] = df["LOT AREA (sqm)"] * df["FLOOR AREA (sqm)"]
    df["IS_LARGE_PROPERTY"] = (df["LOT AREA (sqm)"] >= lot_area_q75).astype(int)

    return df.reindex(columns=feature_cols, fill_value=0).astype(float)


def predict_all_models(df_aligned: pd.DataFrame, scaler, models: dict) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        if "Ridge" in name or "Lasso" in name:
            X_in = scaler.transform(df_aligned)
        else:
            X_in = df_aligned

        pred_log = float(model.predict(X_in)[0])
        pred_price = float(np.expm1(pred_log))
        rows.append({"Model": name, "Pred_price": pred_price})

    return pd.DataFrame(rows).sort_values("Pred_price").reset_index(drop=True)


# ==========================
# UI
# ==========================
# Check required pkl files exist
required = [
    "feature_cols.pkl", "lot_area_q75.pkl", "scaler.pkl",
    "linear_reg.pkl", "ridge_tuned.pkl", "lasso_tuned.pkl", "rf_tuned.pkl", "gb_tuned.pkl"
]
missing = [f for f in required if not (BASE / f).exists()]
if missing:
    st.error("Missing required files in app folder:\n- " + "\n- ".join(missing))
    st.stop()

feature_cols, lot_area_q75, scaler, models = load_artifacts(BASE)

if "pred_table" not in st.session_state:
    st.session_state.pred_table = None

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")
    lot_area = st.number_input("Lot Area (sqm)", min_value=0.0, value=120.0, step=1.0, format="%.2f")
    floor_area = st.number_input("Floor Area (sqm)", min_value=0.0, value=80.0, step=1.0, format="%.2f")

    cat_inputs = {}
    cols = st.columns(min(3, len(CATEGORY_MAPS))) if len(CATEGORY_MAPS) else []
    for i, (col_name, mapping) in enumerate(CATEGORY_MAPS.items()):
        with cols[i % len(cols)] if cols else st.container():
            labels = list(mapping.keys())
            choice = st.selectbox(col_name, options=labels, index=0)
            cat_inputs[col_name] = int(mapping[choice])

    for k, v in DEFAULT_ENCODED_VALUES.items():
        cat_inputs.setdefault(k, int(v))

    user_input = {
        "LOT AREA (sqm)": float(lot_area),
        "FLOOR AREA (sqm)": float(floor_area),
        **cat_inputs
    }

    predict_clicked = st.button("Predict Minimum Bid Price", use_container_width=True, type="primary")

    if predict_clicked:
        with st.spinner("Predicting..."):
            df_aligned = build_features_from_user(user_input, feature_cols, lot_area_q75)
            st.session_state.pred_table = predict_all_models(df_aligned, scaler, models)

with col2:
    st.subheader("Predictions")
    pred_table = st.session_state.pred_table

    if pred_table is None:
        st.info("Fill in the inputs, then click the button.")
    else:
        p_min = float(pred_table["Pred_price"].min())
        p_max = float(pred_table["Pred_price"].max())
        p_med = float(pred_table["Pred_price"].median())

        style_metric_cards(border_left_color="#1D4ED8", background_color="rgba(255, 255, 255, 0.92)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Max (across models)", f"₱{p_max:,.2f}")
        m2.metric("Median (across models)", f"₱{p_med:,.2f}")
        m3.metric("Min (across models)", f"₱{p_min:,.2f}")

        styled = pred_table.style.format({"Pred_price": "₱{:,.2f}"})
        st.dataframe(styled, use_container_width=True)
