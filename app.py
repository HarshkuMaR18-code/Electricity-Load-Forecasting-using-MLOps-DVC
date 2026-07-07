"""
⚡ Electricity Load Forecasting — Streamlit Dashboard
=====================================================
Deployment front-end for:
https://github.com/HarshkuMaR18-code/Electricity-Load-Forecasting-using-MLOps-DVC
 
It reproduces the exact DVC pipeline stages (data ingestion → preprocessing →
feature engineering) in-memory, then loads models/model.pkl if it exists,
or trains the same XGBoost model on the fly (models/ is gitignored, so this
makes the app work on Streamlit Cloud without committing the model).
 
Run locally:  streamlit run app.py
"""
 
import json
import os
import pickle
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
 
# ----------------------------------------------------------------------------
# Config (mirrors mycode/params.yaml and mycode/src/*)
# ----------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/HarshkuMaR18-code/"
    "Electricity-Load-Forecasting-Dataset/refs/heads/main/continuous%20dataset.csv"
)
LOCAL_DATA = os.environ.get("DATA_PATH", os.path.join("data", "continuous_dataset.csv"))
PARAMS_PATH = os.path.join("mycode", "params.yaml")
MODEL_PATH = os.path.join("models", "model.pkl")
METRICS_PATH = os.path.join("reports", "metrics.json")
 
DROP_COLS = [
    "QV2M_toc", "TQL_toc", "W2M_toc", "QV2M_san",
    "QV2M_dav", "TQL_dav", "W2M_dav", "school",
]
# LabelEncoder sorts alphabetically -> deterministic mapping
TIME_BLOCK_CODE = {"afternoon": 0, "evening": 1, "morning": 2, "night": 3}
 
st.set_page_config(
    page_title="Electricity Load Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown(
    """
    <style>
      div[data-testid="stMetric"] {
          background: linear-gradient(135deg, #1e2a3a 0%, #16202e 100%);
          border: 1px solid #2c3e50;
          border-radius: 12px;
          padding: 14px 18px;
      }
      div[data-testid="stMetric"] label { color: #9fb3c8; }
      .block-container { padding-top: 1.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ----------------------------------------------------------------------------
# Pipeline stages (identical logic to mycode/src)
# ----------------------------------------------------------------------------
def load_params() -> dict:
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            return yaml.safe_load(f)
    # fallback defaults = repo params.yaml
    return {
        "data_ingestion": {"test_size": 0.30},
        "feature_engineering": {"n_hours": 2, "m_days": 1},
    }
 
 
def time_block(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    return "night"
 
 
@st.cache_data(show_spinner="📥 Stage 1/4 — Ingesting dataset from GitHub ...")
def ingest() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """data_ingestion.py: load csv, drop columns, sequential train/test split."""
    params = load_params()
    test_size = params["data_ingestion"]["test_size"]
 
    source = LOCAL_DATA if os.path.exists(LOCAL_DATA) else DATA_URL
    df = pd.read_csv(source)
    dt = pd.to_datetime(df["datetime"])          # kept aside for plotting
    df = df.drop(columns=DROP_COLS)
 
    n = len(df)
    train_size = int(n * (1 - test_size))
    return (
        df.iloc[:train_size].copy(), df.iloc[train_size:].copy(),
        dt.iloc[:train_size], dt.iloc[train_size:],
    )
 
 
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """data_preprocessing.py: hour + encoded time block, drop datetime/Holiday_ID."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["time_block_encoded"] = df["hour"].apply(lambda h: TIME_BLOCK_CODE[time_block(h)])
    return df.drop(columns=["datetime", "Holiday_ID"])
 
 
def add_lags(df: pd.DataFrame, n_hours: int, m_days: int) -> pd.DataFrame:
    """feature_engineering.py: hour lags, day lags, target column."""
    df = df.copy()
    for i in range(1, n_hours + 1):
        df[f"hour_lag_{i}"] = df["nat_demand"].shift(i)
    for i in range(1, m_days + 1):
        df[f"day_lag_{i}"] = df["nat_demand"].shift(i * 24)
    df["target"] = df["nat_demand"]
    return df.dropna()
 
 
@st.cache_resource(show_spinner="🤖 Stage 4/4 — Loading / training XGBoost model ...")
def get_model(X_train: np.ndarray, y_train: np.ndarray):
    """model_building.py: load models/model.pkl if present, else train identically."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f), "loaded from models/model.pkl"
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model, "trained in-app (models/ is gitignored)"
 
 
@st.cache_data(show_spinner="⚙️ Stages 2-3/4 — Preprocessing & feature engineering ...")
def build_datasets():
    params = load_params()
    fe = params["feature_engineering"]
 
    train_raw, test_raw, dt_train, dt_test = ingest()
    train_df = add_lags(preprocess(train_raw), fe["n_hours"], fe["m_days"])
    test_df = add_lags(preprocess(test_raw), fe["n_hours"], fe["m_days"])
 
    dt_test = dt_test.loc[test_df.index]
 
    y_train = train_df["nat_demand"].values
    X_train = train_df.drop(columns=["nat_demand"])
    y_test = test_df["nat_demand"].values
    X_test = test_df.drop(columns=["nat_demand"])
    return X_train, y_train, X_test, y_test, dt_test, params
 
 
# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
st.title("⚡ Electricity Load Forecasting")
st.caption(
    "AI-based smart electricity demand prediction · XGBoost · MLOps pipeline with "
    "DVC & DVCLive · [View source on GitHub]"
    "(https://github.com/HarshkuMaR18-code/Electricity-Load-Forecasting-using-MLOps-DVC)"
)
 
X_train, y_train, X_test, y_test, dt_test, params = build_datasets()
model, model_source = get_model(X_train.values, y_train)
 
y_pred = model.predict(X_test.values)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
# ---- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    window = st.slider("Hours shown in forecast chart", 48, len(y_test), 24 * 14, step=24)
    st.divider()
    st.subheader("Pipeline parameters")
    st.json(params)
    st.caption(f"Model: {model_source}")
    st.divider()
    st.markdown(
        "**Pipeline stages**\n\n"
        "1️⃣ Data ingestion\n\n2️⃣ Preprocessing\n\n"
        "3️⃣ Feature engineering\n\n4️⃣ Model training\n\n5️⃣ Evaluation"
    )
 
# ---- KPI cards -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{mae:.2f} MW")
c2.metric("RMSE", f"{np.sqrt(mse):.2f} MW")
c3.metric("R² score", f"{r2:.4f}")
c4.metric("Test samples", f"{len(y_test):,}")
 
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast", "📊 Error analysis", "🧠 Feature importance", "🗃️ Data"]
)
 
# ---- Tab 1: Actual vs predicted -------------------------------------------
with tab1:
    view = slice(-window, None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dt_test.iloc[view], y=y_test[view],
        name="Actual demand", line=dict(color="#4cc9f0", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dt_test.iloc[view], y=y_pred[view],
        name="Predicted demand", line=dict(color="#f72585", width=2, dash="dot"),
    ))
    fig.update_layout(
        height=480, template="plotly_dark",
        title=f"Actual vs predicted national demand — last {window} hours of test set",
        xaxis_title="Time", yaxis_title="Demand (MW)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
 
# ---- Tab 2: Errors ----------------------------------------------------------
with tab2:
    residuals = y_test - y_pred
    e1, e2 = st.columns(2)
    with e1:
        fig_h = px.histogram(
            residuals, nbins=60, template="plotly_dark",
            title="Residual distribution (MW)", labels={"value": "Error"},
        )
        fig_h.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig_h, use_container_width=True)
    with e2:
        fig_s = px.scatter(
            x=y_test, y=y_pred, template="plotly_dark", opacity=0.35,
            title="Predicted vs actual", labels={"x": "Actual (MW)", "y": "Predicted (MW)"},
        )
        lo, hi = float(min(y_test)), float(max(y_test))
        fig_s.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], name="Perfect fit",
                                   line=dict(color="#f72585", dash="dash")))
        fig_s.update_layout(height=420)
        st.plotly_chart(fig_s, use_container_width=True)
 
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            st.caption(f"Last `dvc repro` metrics (reports/metrics.json): `{json.load(f)}`")
 
# ---- Tab 3: Feature importance ---------------------------------------------
with tab3:
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
    fig_i = px.bar(
        imp, orientation="h", template="plotly_dark",
        title="XGBoost feature importance", labels={"value": "Importance", "index": ""},
    )
    fig_i.update_layout(showlegend=False, height=480)
    st.plotly_chart(fig_i, use_container_width=True)
 
# ---- Tab 4: Data ------------------------------------------------------------
with tab4:
    st.markdown("**Processed test set (model input)** — first 500 rows")
    st.dataframe(X_test.head(500), use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download predictions as CSV",
        pd.DataFrame({"datetime": dt_test.values, "actual": y_test, "predicted": y_pred})
        .to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
    )
 
st.divider()
st.caption("Built with Streamlit · XGBoost · DVC — Electricity Load Forecasting using MLOps & DVC")
 