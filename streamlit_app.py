"""
MetroPT-3 Predictive Maintenance Dashboard — Streamlit
Industrial SCADA dark theme — seamless redesign.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MetroPT-3 | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .block-container,
[data-testid="stVerticalBlock"] {
    background-color: #0a0e1a !important;
    color: #e5e7eb !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* kill white topbar */
[data-testid="stHeader"] { background: #0a0e1a !important; border-bottom: 1px solid #1f2937 !important; }
header[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080c17 !important;
    border-right: 1px solid #1a2235 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label { font-family: 'Rajdhani', sans-serif !important; font-size: 15px !important; font-weight: 600 !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }

/* ── Main padding ── */
.block-container { padding: 2rem 2.5rem 2rem 2.5rem !important; max-width: 100% !important; }

/* ── Headings ── */
h1, h2, h3, h4 { font-family: 'Rajdhani', sans-serif !important; color: #f1f5f9 !important; letter-spacing: 0.5px !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0f1624 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #3b82f6 !important; }
[data-testid="stMetricLabel"]  { font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; color: #64748b !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"]  { font-family: 'JetBrains Mono', monospace !important; font-size: 28px !important; font-weight: 600 !important; color: #f1f5f9 !important; }
[data-testid="stMetricDelta"]  { font-family: 'Rajdhani', sans-serif !important; font-size: 13px !important; font-weight: 700 !important; }

/* ── Sliders ── */
[data-testid="stSlider"] { margin-bottom: 6px !important; }
[data-testid="stSlider"] label p { font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px !important; }
[data-testid="stSlider"] > div > div > div { background: #1e293b !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: #3b82f6 !important; border: 2px solid #60a5fa !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] { background: #1e40af !important; color: #e0f2fe !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }

/* ── Checkboxes ── */
[data-testid="stCheckbox"] label p { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; color: #94a3b8 !important; }
[data-testid="stCheckbox"] [data-baseweb="checkbox"] div { border-color: #334155 !important; }
[data-testid="stCheckbox"] [data-baseweb="checkbox"] [data-checked="true"] { background: #2563eb !important; border-color: #2563eb !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    padding: 10px 28px !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover { opacity: 0.9 !important; box-shadow: 0 0 28px rgba(37,99,235,0.5) !important; }

/* secondary button */
.stButton > button[kind="secondary"] {
    background: #0f1624 !important;
    border: 1px solid #1e293b !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #0f1624 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    gap: 4px !important;
    padding: 4px !important;
}
[data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #64748b !important;
    border-radius: 7px !important;
    padding: 6px 18px !important;
    border: none !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1e293b !important;
    color: #3b82f6 !important;
    border-bottom: none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0f1624 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] { border: 1px solid #1e293b !important; border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] th { background: #0f1624 !important; color: #64748b !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stDataFrame"] td { color: #cbd5e1 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; border-left-width: 4px !important; font-family: 'Rajdhani', sans-serif !important; font-size: 15px !important; font-weight: 600 !important; }

/* ── Divider ── */
hr { border-color: #1e293b !important; margin: 20px 0 !important; }

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] p { color: #475569 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }

/* ── Markdown text ── */
[data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* ── Section headers ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
}
/* ── Live risk pill ── */
.risk-pill {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 999px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
}
.risk-low  { background: #052e16; color: #10b981; border: 1px solid #10b981; }
.risk-med  { background: #451a03; color: #f59e0b; border: 1px solid #f59e0b; }
.risk-high { background: #450a0a; color: #ef4444; border: 1px solid #ef4444; animation: pulse 1.5s infinite; }
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
}
/* ── Sidebar branding ── */
.brand-block {
    padding: 28px 20px 20px 20px;
    border-bottom: 1px solid #1a2235;
    margin-bottom: 16px;
}
.brand-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #3b82f6;
    letter-spacing: 2px;
    line-height: 1;
}
.brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #334155;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}
.status-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
}
.dot-green { background: #10b981; box-shadow: 0 0 6px #10b981; }
.dot-yellow { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─── Load models ─────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "saved_models")

@st.cache_resource(show_spinner=False)
def load_models():
    features     = joblib.load(os.path.join(BASE, "features.pkl"))
    rf_model     = joblib.load(os.path.join(BASE, "rf_model.pkl"))
    rf_threshold = float(joblib.load(os.path.join(BASE, "rf_threshold.pkl")))
    sample_input = joblib.load(os.path.join(BASE, "sample_input.pkl"))
    xgb_model, xgb_threshold = None, 0.5
    try:
        import xgboost as xgb_lib
        m = xgb_lib.XGBClassifier()
        json_path = os.path.join(BASE, "xgb_model.json")
        pkl_path  = os.path.join(BASE, "xgb_model.pkl")
        if os.path.exists(json_path):
            m.load_model(json_path)
        elif os.path.exists(pkl_path):
            m = joblib.load(pkl_path)
        else:
            raise FileNotFoundError("No XGBoost model file found")
        xgb_model     = m
        xgb_threshold = float(joblib.load(os.path.join(BASE, "xgb_threshold.pkl")))
    except Exception:
        pass
    return features, rf_model, rf_threshold, xgb_model, xgb_threshold, sample_input

features, rf_model, rf_threshold, xgb_model, xgb_threshold, sample_input = load_models()
DEFAULTS = sample_input.iloc[0].to_dict()

# ─── Plotly dark layout defaults ─────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#0f1624",
    plot_bgcolor="#0f1624",
    font=dict(color="#94a3b8", family="JetBrains Mono"),
)

def dark_axes(fig, xsuffix="", ysuffix=""):
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#1e293b", ticksuffix=xsuffix)
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#1e293b", ticksuffix=ysuffix)
    return fig

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-block">
        <div class="brand-name">MetroPT-3</div>
        <div class="brand-sub">Predictive Maintenance System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Predict", "Model Performance", "Feature Importance"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    rf_dot   = '<span class="status-dot dot-green"></span>'
    xgb_dot  = f'<span class="status-dot {"dot-green" if xgb_model else "dot-yellow"}"></span>'
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#475569;line-height:2">
        {rf_dot}<span style="color:#64748b">Random Forest</span> &nbsp;online<br>
        {xgb_dot}<span style="color:#64748b">XGBoost</span> &nbsp;{'online' if xgb_model else '<span style="color:#f59e0b">offline — brew install libomp</span>'}<br>
        <span style="color:#334155">Features &nbsp;&nbsp;&nbsp;&nbsp;</span><span style="color:#3b82f6">30</span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def infer(vals):
    row = {f: vals.get(f, DEFAULTS.get(f, 0.0)) for f in features}
    X = pd.DataFrame([row])
    rf_prob = float(rf_model.predict_proba(X)[:, 1][0])
    rf_pred = int(rf_prob >= rf_threshold)
    if xgb_model:
        xgb_prob = float(xgb_model.predict_proba(X)[:, 1][0])
        xgb_pred = int(xgb_prob >= xgb_threshold)
        ens_prob = (rf_prob + xgb_prob) / 2
    else:
        xgb_prob = xgb_pred = None
        ens_prob = rf_prob
    ens_pred = int(ens_prob >= 0.5)
    return rf_prob, rf_pred, xgb_prob, xgb_pred, ens_prob, ens_pred

def risk_color(p):
    if p < 0.35: return "#10b981"
    if p < 0.65: return "#f59e0b"
    return "#ef4444"

def risk_label(p):
    if p < 0.35: return "LOW RISK", "risk-low"
    if p < 0.65: return "MODERATE", "risk-med"
    return "FAILURE RISK", "risk-high"

def gauge(value, title, height=260):
    color = risk_color(value / 100)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(size=38, color="#f1f5f9", family="JetBrains Mono")),
        title=dict(text=title, font=dict(size=13, color="#64748b", family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#1e293b", tickfont=dict(color="#334155", size=10)),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#0a0e1a",
            bordercolor="#1e293b",
            borderwidth=1,
            steps=[
                dict(range=[0, 35],  color="#05140d"),
                dict(range=[35, 65], color="#190f02"),
                dict(range=[65, 100], color="#1a0404"),
            ],
            threshold=dict(line=dict(color=color, width=2), thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(**DARK_LAYOUT, height=height, margin=dict(t=30, b=0, l=20, r=20))
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Predict":
    # Page header
    st.markdown('<h1 style="font-size:34px;margin-bottom:2px;">Failure Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;font-size:12px;letter-spacing:1px;margin-bottom:28px;">ADJUST SENSOR READINGS — REAL-TIME RISK SCORING</p>', unsafe_allow_html=True)

    SENSOR_FIELDS = [
        ("TP2",           "TP2 — Inlet Pressure",      -1.0, 12.0, 0.01, "bar"),
        ("TP3",           "TP3 — Discharge Pressure",  -1.0, 12.0, 0.01, "bar"),
        ("H1",            "H1 — Line Pressure",        -1.0, 12.0, 0.01, "bar"),
        ("DV_pressure",   "DV Pressure",               -1.0,  5.0, 0.01, "bar"),
        ("Reservoirs",    "Reservoirs",                -1.0, 12.0, 0.01, "bar"),
        ("Oil_temperature","Oil Temperature",            0.0,100.0,  0.1, "°C"),
        ("Motor_current", "Motor Current",               0.0, 10.0, 0.01, "A"),
    ]
    BINARY_FIELDS = [
        ("COMP",            "COMP"),
        ("DV_eletric",      "DV Electric"),
        ("Towers",          "Towers"),
        ("MPG",             "MPG"),
        ("LPS",             "LPS"),
        ("Pressure_switch", "Pres. Switch"),
        ("Oil_level",       "Oil Level"),
    ]
    DERIVED = [
        "time_diff","Motor_current_mean_5","Motor_current_std_5",
        "Oil_temperature_mean_5","Oil_temperature_std_5","TP2_mean_5","TP3_mean_5",
        "Motor_current_diff","Oil_temperature_diff","TP2_diff","TP3_diff",
        "Motor_current_lag1","Motor_current_lag2","Oil_temperature_lag1","Oil_temperature_lag2",
    ]

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-label">Sensor Inputs</div>', unsafe_allow_html=True)
        vals = {}
        c1, c2 = st.columns(2, gap="medium")
        for i, (key, label, mn, mx, step, unit) in enumerate(SENSOR_FIELDS):
            col = c1 if i % 2 == 0 else c2
            with col:
                vals[key] = st.slider(
                    f"{label}  [{unit}]",
                    min_value=float(mn), max_value=float(mx), step=float(step),
                    value=float(round(DEFAULTS.get(key, 0.0), 2)),
                    key=f"s_{key}",
                )

        vals["Caudal_impulses"] = st.slider(
            "Caudal Impulses",
            min_value=0, max_value=5, step=1,
            value=int(DEFAULTS.get("Caudal_impulses", 1)),
            key="s_Caudal",
        )

        st.markdown('<div class="section-label" style="margin-top:20px;">Digital Signals</div>', unsafe_allow_html=True)
        bcols = st.columns(4, gap="small")
        for i, (key, label) in enumerate(BINARY_FIELDS):
            with bcols[i % 4]:
                vals[key] = float(st.checkbox(label, value=bool(DEFAULTS.get(key, 0)), key=f"cb_{key}"))

        for key in DERIVED:
            vals[key] = DEFAULTS.get(key, 0.0)

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1.2, 1, 3])
        with b1:
            run = st.button("▶  Run", use_container_width=True)
        with b2:
            reset = st.button("↺  Reset", use_container_width=True)

        if reset:
            st.rerun()

    with right:
        st.markdown('<div class="section-label">Live Risk Monitor</div>', unsafe_allow_html=True)

        # Always show live gauge (updates on every slider move)
        rf_p, rf_pred, xgb_p, xgb_pred, ens_p, ens_pred = infer(vals)
        label_text, label_cls = risk_label(ens_p)

        st.markdown(f'<div style="text-align:center;margin-bottom:12px"><span class="risk-pill {label_cls}">{label_text}</span></div>', unsafe_allow_html=True)
        st.plotly_chart(gauge(round(ens_p * 100, 1), "ENSEMBLE RISK"), use_container_width=True)

        # Mini gauges row
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(gauge(round(rf_p * 100, 1), "RANDOM FOREST", height=190), use_container_width=True)
        with g2:
            xgb_val = round(xgb_p * 100, 1) if xgb_p is not None else round(rf_p * 100, 1)
            st.plotly_chart(gauge(xgb_val, "XGBOOST" if xgb_model else "XGBOOST (offline)", height=190), use_container_width=True)

        # Threshold progress bars
        st.markdown('<div class="section-label" style="margin-top:8px;">Model Thresholds</div>', unsafe_allow_html=True)

        def threshold_bar(label, prob, threshold, color):
            pct = prob * 100
            tpct = threshold * 100
            bar_color = risk_color(prob)
            st.markdown(f"""
            <div style="margin-bottom:14px">
                <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:5px">
                    <span style="color:#64748b;font-family:'JetBrains Mono',monospace">{label}</span>
                    <span style="color:{bar_color};font-family:'JetBrains Mono',monospace;font-weight:600">{pct:.1f}%</span>
                </div>
                <div style="background:#0a0e1a;border:1px solid #1e293b;border-radius:4px;height:8px;position:relative;overflow:visible">
                    <div style="width:{min(pct,100):.1f}%;height:100%;background:{bar_color};border-radius:4px;transition:width 0.3s ease"></div>
                    <div style="position:absolute;top:-3px;left:{tpct:.1f}%;width:2px;height:14px;background:#ef4444;border-radius:1px" title="Threshold {tpct:.1f}%"></div>
                </div>
                <div style="font-size:10px;color:#334155;font-family:'JetBrains Mono',monospace;margin-top:3px">threshold: {tpct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        threshold_bar("Random Forest", rf_p, rf_threshold, "#3b82f6")
        if xgb_p is not None:
            threshold_bar("XGBoost", xgb_p, xgb_threshold, "#f59e0b")

    # ── Result detail (only after Run) ──
    if run:
        st.markdown("---")
        st.markdown('<div class="section-label">Prediction Detail</div>', unsafe_allow_html=True)

        rows = [
            {"Model": "Random Forest", "Probability": f"{rf_p*100:.2f}%",
             "Threshold": f"{rf_threshold*100:.1f}%",
             "Result": "FAILURE" if rf_pred else "NORMAL"},
        ]
        if xgb_p is not None:
            rows.append({"Model": "XGBoost", "Probability": f"{xgb_p*100:.2f}%",
                         "Threshold": f"{xgb_threshold*100:.1f}%",
                         "Result": "FAILURE" if xgb_pred else "NORMAL"})
        rows.append({"Model": "Ensemble", "Probability": f"{ens_p*100:.2f}%",
                     "Threshold": "50.0%",
                     "Result": "FAILURE" if ens_pred else "NORMAL"})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if ens_pred:
            st.error("FAILURE IMMINENT — Ensemble predicts compressor failure. Inspect equipment immediately.", icon="🚨")
        else:
            st.success("NORMAL OPERATION — All models indicate healthy compressor state.", icon="✅")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<h1 style="font-size:34px;margin-bottom:2px;">Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;font-size:12px;letter-spacing:1px;margin-bottom:28px;">TEST SET EVALUATION — METROPT-3 DATASET</p>', unsafe_allow_html=True)

    MODELS = [
        {"name": "Random Forest", "accuracy": 0.9371, "precision": 0.7521, "recall": 0.9074, "f1": 0.8224, "roc_auc": 0.9771, "color": "#3b82f6"},
        {"name": "XGBoost",       "accuracy": 0.9579, "precision": 0.8929, "recall": 0.8384, "f1": 0.8648, "roc_auc": 0.9711, "color": "#f59e0b"},
        {"name": "LSTM",          "accuracy": 0.8291, "precision": 0.4834, "recall": 0.6286, "f1": 0.5465, "roc_auc": 0.8448, "color": "#10b981"},
        {"name": "Transformer",   "accuracy": 0.7978, "precision": 0.4221, "recall": 0.6343, "f1": 0.5068, "roc_auc": 0.8081, "color": "#8b5cf6"},
    ]

    # KPI strip
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1: st.metric("Best F1",       f"{max(m['f1'] for m in MODELS)*100:.2f}%",       delta="XGBoost")
    with k2: st.metric("Best ROC AUC",  f"{max(m['roc_auc'] for m in MODELS)*100:.2f}%",  delta="Random Forest")
    with k3: st.metric("Best Accuracy", f"{max(m['accuracy'] for m in MODELS)*100:.2f}%", delta="XGBoost")
    with k4: st.metric("Best Precision",f"{max(m['precision'] for m in MODELS)*100:.2f}%",delta="XGBoost")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Metrics Table", "Bar Comparison", "Radar Chart"])

    with tab1:
        df = pd.DataFrame(MODELS)
        for col in ["accuracy","f1","precision","recall","roc_auc"]:
            df[col] = df[col].apply(lambda v: f"{v*100:.2f}%")
        df.columns = ["Model","Color","Accuracy","F1","Precision","Recall","ROC AUC"]
        st.dataframe(df[["Model","Accuracy","F1","Precision","Recall","ROC AUC"]], use_container_width=True, hide_index=True)
        st.caption("RF and XGB run live inference. LSTM and Transformer are offline evaluation benchmarks only.")

    with tab2:
        metric = st.selectbox("Metric", ["f1","accuracy","precision","recall","roc_auc"],
                              format_func=lambda x: {"f1":"F1 Score","accuracy":"Accuracy","precision":"Precision","recall":"Recall","roc_auc":"ROC AUC"}[x],
                              key="bar_metric")
        fig = go.Figure(go.Bar(
            x=[m["name"] for m in MODELS],
            y=[m[metric] * 100 for m in MODELS],
            marker=dict(color=[m["color"] for m in MODELS], line=dict(color="#0a0e1a", width=1)),
            text=[f"{m[metric]*100:.2f}%" for m in MODELS],
            textposition="outside",
            textfont=dict(color="#94a3b8", family="JetBrains Mono", size=12),
        ))
        fig.update_layout(**DARK_LAYOUT, height=360,
                          yaxis=dict(range=[0,108], gridcolor="#1e293b", ticksuffix="%", tickfont=dict(size=11)),
                          xaxis=dict(tickfont=dict(size=13, family="Rajdhani")),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cats = ["Accuracy","F1","Precision","Recall","ROC AUC"]
        fig = go.Figure()
        for m in MODELS:
            vals_r = [m["accuracy"], m["f1"], m["precision"], m["recall"], m["roc_auc"]]
            vals_r += [vals_r[0]]
            fig.add_trace(go.Scatterpolar(
                r=[v * 100 for v in vals_r],
                theta=cats + [cats[0]],
                fill="toself",
                name=m["name"],
                line=dict(color=m["color"], width=2),
                fillcolor=m["color"],
                opacity=0.12,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="#0a0e1a",
                angularaxis=dict(tickfont=dict(color="#64748b", family="JetBrains Mono", size=11), linecolor="#1e293b", gridcolor="#1e293b"),
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#334155", size=10), gridcolor="#1e293b", tickformat=".0f", ticksuffix="%"),
            ),
            paper_bgcolor="#0f1624",
            legend=dict(font=dict(color="#94a3b8", family="Rajdhani", size=13), bgcolor="#0f1624", bordercolor="#1e293b", borderwidth=1),
            height=460,
            margin=dict(t=30, b=30, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.markdown('<h1 style="font-size:34px;margin-bottom:2px;">Feature Importance</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;font-size:12px;letter-spacing:1px;margin-bottom:28px;">TOP FEATURES BY CONTRIBUTION TO FAILURE PREDICTION</p>', unsafe_allow_html=True)

    rf_imp  = dict(zip(features, rf_model.feature_importances_.tolist()))
    rf_top  = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:10]

    has_xgb = xgb_model and hasattr(xgb_model, "feature_importances_")
    if has_xgb:
        xgb_imp = dict(zip(features, xgb_model.feature_importances_.tolist()))
        xgb_top = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:10]

    def h_bar(data, title, color, height=420):
        feats = [d[0] for d in data][::-1]
        imps  = [d[1] * 100 for d in data][::-1]
        fig = go.Figure(go.Bar(
            x=imps, y=feats, orientation="h",
            marker=dict(
                color=imps,
                colorscale=[[0, "#0f1624"], [0.4, color], [1, "#ffffff"]],
                line=dict(color="#0a0e1a", width=0.5),
            ),
            text=[f"{v:.2f}%" for v in imps],
            textposition="outside",
            textfont=dict(color="#64748b", family="JetBrains Mono", size=11),
        ))
        fig.update_layout(**DARK_LAYOUT,
            title=dict(text=title, font=dict(size=14, color="#94a3b8", family="JetBrains Mono")),
            height=height,
            margin=dict(t=44, b=10, l=10, r=70),
            xaxis=dict(gridcolor="#1e293b", tickformat=".1f", ticksuffix="%", tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=12, family="JetBrains Mono", color="#94a3b8")),
            showlegend=False,
        )
        return fig

    if has_xgb:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.plotly_chart(h_bar(rf_top,  "Random Forest — Mean Decrease Impurity", "#3b82f6"), use_container_width=True)
        with c2:
            st.plotly_chart(h_bar(xgb_top, "XGBoost — Gain", "#f59e0b"), use_container_width=True)
    else:
        st.plotly_chart(h_bar(rf_top, "Random Forest — Mean Decrease Impurity", "#3b82f6"), use_container_width=True)

    # Rankings table
    st.markdown('<div class="section-label" style="margin-top:8px;">Full Rankings</div>', unsafe_allow_html=True)
    rf_df = pd.DataFrame(rf_top, columns=["Feature", "RF Importance"])
    rf_df["Rank"] = range(1, len(rf_df) + 1)
    rf_df["RF Importance"] = rf_df["RF Importance"].apply(lambda v: f"{v*100:.4f}%")
    if has_xgb:
        xgb_df = pd.DataFrame(xgb_top, columns=["Feature", "XGB Importance"])
        xgb_df["XGB Importance"] = xgb_df["XGB Importance"].apply(lambda v: f"{v*100:.4f}%")
        merged = rf_df.merge(xgb_df, on="Feature", how="outer").fillna("—")
        st.dataframe(merged[["Rank","Feature","RF Importance","XGB Importance"]], use_container_width=True, hide_index=True)
    else:
        st.dataframe(rf_df[["Rank","Feature","RF Importance"]], use_container_width=True, hide_index=True)

    st.caption("RF = mean decrease in impurity  |  XGB = gain (contribution to loss reduction)")
