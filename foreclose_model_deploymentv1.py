import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go

st.set_page_config(page_title="Minimum Bid Price Predictor", layout="wide")

# ==========================
# Global CSS
# ==========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    .stApp { background: #F6F8FF; font-family: "Poppins","Inter","Segoe UI",Arial,sans-serif; }

    .title-banner {
        background: linear-gradient(90deg, #0F172A 0%, #1D4ED8 60%, #0EA5E9 100%);
        padding: 18px 24px; border-radius: 14px; color: white;
        font-size: 30px; font-weight: 800; margin-bottom: 16px;
        box-shadow: 0 10px 26px rgba(2, 6, 23, 0.22);
    }

    .info-box {
        background: #EFF6FF; border-left: 4px solid #1D4ED8;
        border-radius: 8px; padding: 12px 16px;
        font-size: 0.88rem; color: #1E3A5F; margin-bottom: 10px;
    }

    .warn-box {
        background: #FFFBEB; border-left: 4px solid #F59E0B;
        border-radius: 8px; padding: 10px 14px;
        font-size: 0.85rem; color: #92400E; margin-bottom: 8px;
    }

    .error-box {
        background: #FEF2F2; border-left: 4px solid #EF4444;
        border-radius: 8px; padding: 10px 14px;
        font-size: 0.85rem; color: #991B1B; margin-bottom: 8px;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0F172A 0%, #1D4ED8 60%, #0EA5E9 100%) !important;
        border: 0 !important; color: #FFF !important; font-weight: 800 !important;
        border-radius: 12px !important; padding: 0.70rem 1rem !important;
        box-shadow: 0 12px 26px rgba(2, 6, 23, 0.22) !important;
        transition: transform .08s, box-shadow .08s;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(2, 6, 23, 0.28) !important;
        filter: saturate(1.05);
    }

    label[data-testid="stMetricLabel"] p { font-size: 0.95rem !important; color: rgba(15,23,42,.85) !important; }
    div[data-testid="stMetricValue"] p { font-size: 1.05rem !important; color: #0F172A !important; }
    </style>

    <div class="title-banner">🏠 Foreclosed Property – Minimum Bid Price Predictor</div>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Constants & Validation Limits
# ==========================
CATEGORY_MAPS = {
    "REGION":  {"METRO MANILA": 0, "NON-METRO MANILA": 1},
    "REMARKS": {"TCT UNDER THE BANK": 0, "FOR TITLE CONSOLIDATION": 1},
    "STATUS":  {"UNOCCUPIED": 0, "OCCUPIED": 1},
}
DEFAULT_ENCODED_VALUES = {}

MAX_LOT_AREA    = 1_000.0   # ← updated to 1000
MAX_FLOOR_AREA  = 1_000.0   # ← updated to 1000
WARN_LOT_AREA   = 800.0
WARN_FLOOR_AREA = 800.0

BASE = Path(__file__).resolve().parent

# ==========================
# Artifact Loader
# ==========================
@st.cache_resource
def load_artifacts(base_dir: Path):
    feature_cols  = joblib.load(base_dir / "feature_cols.pkl")
    lot_area_q75  = joblib.load(base_dir / "lot_area_q75.pkl")
    scaler        = joblib.load(base_dir / "scaler.pkl")
    models = {
        "Linear Reg":                joblib.load(base_dir / "linear_reg.pkl"),
        "Ridge (Tuned)":             joblib.load(base_dir / "ridge_tuned.pkl"),
        "Lasso (Tuned)":             joblib.load(base_dir / "lasso_tuned.pkl"),
        "Random Forest (Tuned)":     joblib.load(base_dir / "rf_tuned.pkl"),
        "Gradient Boosting (Tuned)": joblib.load(base_dir / "gb_tuned.pkl"),
    }
    return feature_cols, lot_area_q75, scaler, models


# ==========================
# Input Validation
# ==========================
def validate_inputs(lot_area: float, floor_area: float) -> tuple[list[str], list[str]]:
    errors, warnings = [], []

    if lot_area <= 0:
        errors.append("Lot Area must be greater than 0 sqm.")
    if floor_area <= 0:
        errors.append("Floor Area must be greater than 0 sqm.")
    if lot_area > MAX_LOT_AREA:
        errors.append(
            f"Lot Area ({lot_area:,.0f} sqm) exceeds the maximum allowed value "
            f"of {MAX_LOT_AREA:,.0f} sqm. Please check your input."
        )
    if floor_area > MAX_FLOOR_AREA:
        errors.append(
            f"Floor Area ({floor_area:,.0f} sqm) exceeds the maximum allowed value "
            f"of {MAX_FLOOR_AREA:,.0f} sqm. Please check your input."
        )
    if not errors and floor_area > lot_area:
        errors.append(
            f"Floor Area ({floor_area:,.2f} sqm) cannot exceed Lot Area "
            f"({lot_area:,.2f} sqm). Please review your values."
        )

    if not errors:
        if lot_area > WARN_LOT_AREA:
            warnings.append(
                f"Lot Area ({lot_area:,.0f} sqm) is unusually large. "
                "Verify the value before predicting."
            )
        if floor_area > WARN_FLOOR_AREA:
            warnings.append(
                f"Floor Area ({floor_area:,.0f} sqm) is unusually large. "
                "Verify the value before predicting."
            )
        ratio = floor_area / lot_area if lot_area > 0 else 0
        if ratio > 0.95:
            warnings.append(
                f"Floor-to-lot ratio is {ratio:.0%}. "
                "Very high coverage ratios are rare — double-check your inputs."
            )

    return errors, warnings


# ==========================
# Feature Engineering
# ==========================
def build_features_from_user(
    user_input: dict, feature_cols, lot_area_q75
) -> pd.DataFrame:
    df = pd.DataFrame([user_input]).copy()

    for col in ["LOT AREA (sqm)", "FLOOR AREA (sqm)"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    lot_area_orig = np.expm1(df["LOT AREA (sqm)"])
    floor_orig    = np.expm1(df["FLOOR AREA (sqm)"])

    df["TOTAL_AREA"]         = df["LOT AREA (sqm)"] + df["FLOOR AREA (sqm)"]
    df["FLOOR_TO_LOT_RATIO"] = np.log1p(floor_orig / (lot_area_orig + 1))
    df["LOG_LOT_x_FLOOR"]    = df["LOT AREA (sqm)"] * df["FLOOR AREA (sqm)"]
    df["IS_LARGE_PROPERTY"]  = (df["LOT AREA (sqm)"] >= lot_area_q75).astype(int)

    return df.reindex(columns=feature_cols, fill_value=0).astype(float)


# ==========================
# Prediction
# ==========================
def predict_all_models(
    df_aligned: pd.DataFrame, scaler, models: dict
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        X_in = scaler.transform(df_aligned) if ("Ridge" in name or "Lasso" in name) else df_aligned
        pred_log   = float(model.predict(X_in)[0])
        pred_price = float(np.expm1(pred_log))
        rows.append({"Model": name, "Pred_price": pred_price})
    return pd.DataFrame(rows).sort_values("Pred_price").reset_index(drop=True)


# ==========================
# Chart Helper
# ==========================
MODEL_COLOR_MAP = {
    "Linear Reg":                "#6366F1",
    "Ridge (Tuned)":             "#0EA5E9",
    "Lasso (Tuned)":             "#22D3EE",
    "Random Forest (Tuned)":     "#10B981",
    "Gradient Boosting (Tuned)": "#1D4ED8",
}

def bar_chart_predictions(pred_table: pd.DataFrame) -> go.Figure:
    colors = [MODEL_COLOR_MAP.get(m, "#94A3B8") for m in pred_table["Model"]]
    fig = go.Figure(
        go.Bar(
            x=pred_table["Pred_price"],
            y=pred_table["Model"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"₱{v:,.0f}" for v in pred_table["Pred_price"]],
            textposition="outside",
            textfont=dict(size=12, family="Poppins"),
            hovertemplate="<b>%{y}</b><br>Predicted Price: ₱%{x:,.2f}<extra></extra>",
        )
    )
    p_min = pred_table["Pred_price"].min()
    p_max = pred_table["Pred_price"].max()
    pad   = (p_max - p_min) * 0.22 if p_max > p_min else p_max * 0.25

    fig.update_layout(
        title=dict(text="Model Predictions Comparison", font=dict(size=15, family="Poppins", color="#0F172A")),
        xaxis=dict(
            title="Predicted Minimum Bid Price (₱)",
            tickprefix="₱", tickformat=",.0f",
            range=[max(0, p_min - pad * 0.1), p_max + pad],
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=20, t=45, b=40),
        height=320,
        font=dict(family="Poppins"),
    )
    return fig


# ==========================
# Pre-run checks
# ==========================
required = [
    "feature_cols.pkl", "lot_area_q75.pkl", "scaler.pkl",
    "linear_reg.pkl", "ridge_tuned.pkl", "lasso_tuned.pkl",
    "rf_tuned.pkl", "gb_tuned.pkl",
]
missing = [f for f in required if not (BASE / f).exists()]
if missing:
    st.error("⚠️ Missing required model files in app folder:\n- " + "\n- ".join(missing))
    st.stop()

feature_cols, lot_area_q75, scaler, models = load_artifacts(BASE)

if "pred_table" not in st.session_state:
    st.session_state.pred_table = None
if "last_input" not in st.session_state:
    st.session_state.last_input = None

# ==========================
# Layout
# ==========================
col1, col2 = st.columns([1, 1.4], gap="large")

# ---------- LEFT: Inputs ----------
with col1:
    st.subheader("🔢 Property Inputs")

    st.markdown(
        '<div class="info-box">💡 Enter property details below. '
        "Lot Area ≥ Floor Area. Max for both: 1,000 sqm.</div>",
        unsafe_allow_html=True,
    )

    lot_area = st.number_input(
        "Lot Area (sqm)",
        min_value=1.0,
        max_value=MAX_LOT_AREA,
        value=120.0,
        step=1.0,
        format="%.2f",
        help=f"Valid range: 1 – {MAX_LOT_AREA:,.0f} sqm",
    )
    floor_area = st.number_input(
        "Floor Area (sqm)",
        min_value=1.0,
        max_value=MAX_FLOOR_AREA,
        value=80.0,
        step=1.0,
        format="%.2f",
        help=f"Valid range: 1 – {MAX_FLOOR_AREA:,.0f} sqm. Must not exceed Lot Area.",
    )

    st.markdown("---")

    cat_inputs = {}
    cols_cat = st.columns(min(3, len(CATEGORY_MAPS))) if CATEGORY_MAPS else []
    for i, (col_name, mapping) in enumerate(CATEGORY_MAPS.items()):
        with cols_cat[i % len(cols_cat)] if cols_cat else st.container():
            labels = list(mapping.keys())
            choice = st.selectbox(col_name, options=labels, index=0)
            cat_inputs[col_name] = int(mapping[choice])

    for k, v in DEFAULT_ENCODED_VALUES.items():
        cat_inputs.setdefault(k, int(v))

    user_input = {
        "LOT AREA (sqm)":   float(lot_area),
        "FLOOR AREA (sqm)": float(floor_area),
        **cat_inputs,
    }

    live_errors, live_warns = validate_inputs(lot_area, floor_area)
    for w in live_warns:
        st.markdown(f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True)
    for e in live_errors:
        st.markdown(f'<div class="error-box">🚫 {e}</div>', unsafe_allow_html=True)

    predict_clicked = st.button(
        "🔍 Predict Minimum Bid Price",
        use_container_width=True,
        type="primary",
        disabled=bool(live_errors),
    )

    if predict_clicked and not live_errors:
        with st.spinner("Running models…"):
            df_aligned = build_features_from_user(user_input, feature_cols, lot_area_q75)
            st.session_state.pred_table = predict_all_models(df_aligned, scaler, models)
            st.session_state.last_input = user_input.copy()

# ---------- RIGHT: Results ----------
with col2:
    st.subheader("📊 Prediction Results")
    pred_table = st.session_state.pred_table

    if pred_table is None:
        st.info("Fill in the property details on the left, then click **Predict**.")

        placeholder_fig = go.Figure()
        placeholder_fig.add_annotation(
            text="📈 Your predictions will appear here",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#94A3B8", family="Poppins"),
        )
        placeholder_fig.update_layout(
            plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=260, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(placeholder_fig, use_container_width=True)

    else:
        p_min = float(pred_table["Pred_price"].min())
        p_max = float(pred_table["Pred_price"].max())
        p_med = float(pred_table["Pred_price"].median())
        spread_pct = ((p_max - p_min) / p_med * 100) if p_med > 0 else 0.0

        style_metric_cards(
            border_left_color="#1D4ED8",
            background_color="rgba(255, 255, 255, 0.92)",
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔼 Max",    f"₱{p_max:,.2f}")
        m2.metric("📍 Median", f"₱{p_med:,.2f}")
        m3.metric("🔽 Min",    f"₱{p_min:,.2f}")
        m4.metric("📐 Spread", f"{spread_pct:.1f}%",
                  help="(Max − Min) / Median — model disagreement level")

        if spread_pct < 10:
            badge = "🟢 **High agreement** across models — estimate is reliable."
        elif spread_pct < 30:
            badge = "🟡 **Moderate spread** — consider using the median as reference."
        else:
            badge = "🔴 **High spread** — models disagree; treat estimates as a wide range."
        st.markdown(f"> {badge}")

        # --- Bar chart (full width now that pie is removed) ---
        st.plotly_chart(bar_chart_predictions(pred_table), use_container_width=True)

        # --- Detailed table ---
        with st.expander("📋 Full Prediction Table", expanded=False):
            def highlight_min(row):
                is_min = row["Pred_price"] == pred_table["Pred_price"].min()
                return ["background-color: #D1FAE5; font-weight:700" if is_min else "" for _ in row]

            styled = (
                pred_table.style
                .apply(highlight_min, axis=1)
                .format({"Pred_price": "₱{:,.2f}"})
                .set_properties(**{"font-family": "Poppins", "font-size": "13px"})
            )
            st.dataframe(styled, use_container_width=True)

        # --- Input recap ---
        with st.expander("🔎 Input Used for This Prediction", expanded=False):
            last = st.session_state.last_input
            recap = pd.DataFrame(
                [{"Field": k, "Value": v} for k, v in last.items()]
            )
            st.dataframe(recap, use_container_width=True, hide_index=True)
