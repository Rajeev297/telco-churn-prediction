"""
Customer Churn Analytics Platform
Business-ready churn prediction, bulk analysis, and insights dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import io
import time
from typing import Tuple, List, Optional
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import re
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import DataPreprocessor

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

st.set_page_config(
    page_title="Churn Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ───────────────────────── REQUIRED COLUMNS ─────────────────────────
REQUIRED_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

CATEGORICAL_MAPS = {
    'gender': ['Male', 'Female'],
    'SeniorCitizen': ['0', '1', 0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
}

# ───────────────────────── STYLING ─────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', -apple-system, sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.4rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.5);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    .card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
    }
    .card h3 {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0 0 1.2rem 0;
        letter-spacing: 0.3px;
    }
    .card h4 {
        color: rgba(255,255,255,0.8);
        font-weight: 500;
        font-size: 0.95rem;
        margin: 0 0 0.8rem 0;
    }

    .kpi-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-card .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
    }
    .kpi-card .kpi-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        margin-top: 0.2rem;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    .risk-gauge {
        text-align: center;
        padding: 1rem 0;
    }
    .risk-gauge .percent {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .risk-gauge .label {
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .risk-gauge .sub {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    .prediction-badge {
        display: inline-block;
        padding: 0.3rem 1.5rem;
        border-radius: 40px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }

    .rec-box {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .rec-box p { margin: 0; color: rgba(255,255,255,0.8); font-size: 0.9rem; }

    .shap-bar {
        display: flex;
        align-items: center;
        margin: 0.4rem 0;
    }
    .shap-bar .fname {
        width: 140px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.7);
        text-align: right;
        padding-right: 10px;
        flex-shrink: 0;
    }
    .shap-bar .track {
        flex: 1;
        height: 18px;
        background: rgba(255,255,255,0.06);
        border-radius: 20px;
        position: relative;
        overflow: hidden;
    }
    .shap-bar .fill {
        height: 100%;
        border-radius: 20px;
        transition: width 0.4s;
    }
    .shap-bar .value {
        width: 50px;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-align: left;
        padding-left: 8px;
        flex-shrink: 0;
    }

    div[data-testid="stSlider"] label p { color: rgba(255,255,255,0.7) !important; font-size: 0.9rem; }
    div[data-testid="stSelectbox"] label p { color: rgba(255,255,255,0.7) !important; font-size: 0.9rem; }
    .stSelectbox div[data-baseweb="select"] > div { background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 10px !important; color: #fff !important; }
    .stSlider div[data-baseweb="slider"] { margin-top: 0.5rem; }

    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
        color: white !important;
    }
    .stButton button:active, .stButton button:focus { color: white !important; }

    .stDownloadButton button {
        background: linear-gradient(135deg, #34d399, #059669) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stDownloadButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(52,211,153,0.4);
        color: white !important;
    }

    hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.5rem 0 !important; }

    .footer { text-align: center; padding: 1.5rem 0 2rem 0; color: rgba(255,255,255,0.2); font-size: 0.75rem; }

    .stSpinner > div { border-color: #667eea !important; }

    div[data-testid="stMetric"] { background: rgba(255,255,255,0.06); border-radius: 16px; padding: 1rem; border: 1px solid rgba(255,255,255,0.06); }
    div[data-testid="stMetric"] label { color: rgba(255,255,255,0.5) !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.3px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; }

    .stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2) !important; }

    .stDataFrame { background: transparent !important; }
    .stDataFrame td, .stDataFrame th { color: rgba(255,255,255,0.8) !important; }

    div[data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); }
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255,255,255,0.5) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 10px 10px 0 0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(255,255,255,0.06) !important;
        border-bottom: 2px solid #667eea !important;
    }

    .insight-card {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #667eea;
        margin: 0.6rem 0;
    }
    .insight-card p { margin: 0; color: rgba(255,255,255,0.75); font-size: 0.9rem; }

    .success-msg {
        background: rgba(52,211,153,0.12);
        border: 1px solid rgba(52,211,153,0.2);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        color: #34d399;
        font-weight: 500;
        text-align: center;
    }
    .error-msg {
        background: rgba(248,113,113,0.12);
        border: 1px solid rgba(248,113,113,0.2);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        color: #f87171;
        font-weight: 500;
    }

    @media (max-width: 640px) {
        .main-header h1 { font-size: 1.6rem; }
        .risk-gauge .percent { font-size: 2.5rem; }
        .card { padding: 1.2rem; }
        .kpi-card .kpi-value { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)


# ───────────────────────── LOAD TRAINED METRICS ─────────────────────────
@st.cache_resource
def load_trained_metrics() -> Optional[dict]:
    eval_path = os.path.join(MODELS_DIR, '..', 'reports', 'model_evaluation.txt')
    if not os.path.exists(eval_path):
        return None
    try:
        with open(eval_path, 'r') as f:
            text = f.read()
        m = re.search(r'BEST MODEL:\s*(.+?)\s*$', text, re.MULTILINE)
        if not m:
            return None
        best = m.group(1)
        start = text.index(best + ':')
        rest = text[start:]
        end = rest.index('\n\n') if '\n\n' in rest else len(rest)
        section = rest[:end]
        labels = {'Accuracy': 'Test Accuracy', 'Precision': 'Test Precision', 'Recall': 'Test Recall', 'F1 Score': 'Test F1 Score', 'ROC-AUC': 'Test ROC-AUC'}
        metrics = {}
        for name, label in labels.items():
            match = re.search(rf'{re.escape(label)}:\s+([\d.]+)', section)
            if match:
                metrics[name] = float(match.group(1))
        return metrics or None
    except Exception:
        return None


# ───────────────────────── LOAD ARTIFACTS ─────────────────────────
@st.cache_resource
def load_artifacts():
    model = None
    preprocessor = None
    explainer = None
    threshold = 0.5

    def _is_valid_pickle(path):
        """Check if file is a valid pickle (not a Git LFS pointer)."""
        try:
            with open(path, 'rb') as f:
                header = f.read(100)
                return not header.startswith(b'version https://git-lfs')
        except Exception:
            return False

    try:
        prep_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        if os.path.exists(prep_path) and _is_valid_pickle(prep_path):
            with open(prep_path, 'rb') as f:
                preprocessor = pickle.load(f)
    except Exception as e:
        st.warning(f"Preprocessor not loaded: {e}")
    try:
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl') and 'preprocessor' not in f and 'explainer' not in f and 'metadata' not in f]
        if files:
            latest = max(files, key=lambda f: os.path.getctime(os.path.join(MODELS_DIR, f)))
            path = os.path.join(MODELS_DIR, latest)
            if _is_valid_pickle(path):
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                    if isinstance(obj, dict) and 'model' in obj:
                        model = obj['model']
                        threshold = obj.get('threshold', 0.5)
                    else:
                        model = obj
    except Exception as e:
        st.warning(f"Model not loaded: {e}")
    try:
        exp_path = os.path.join(MODELS_DIR, 'shap_explainer.pkl')
        if os.path.exists(exp_path) and _is_valid_pickle(exp_path):
            with open(exp_path, 'rb') as f:
                explainer = pickle.load(f)
    except Exception:
        pass

    if model is None or preprocessor is None:
        files_exist = any(f.endswith('.pkl') for f in os.listdir(MODELS_DIR)) if os.path.isdir(MODELS_DIR) else False
        if files_exist:
            st.error("Model files are Git LFS pointers. Enable Git LFS in Streamlit Cloud: Settings → Advanced → Git LFS = ON")
    return model, preprocessor, explainer, threshold

model, preprocessor, explainer, MODEL_THRESHOLD = load_artifacts()
trained_metrics = load_trained_metrics()


# ───────────────────────── RISK HELPERS ─────────────────────────
def get_risk(prob: float) -> Tuple[str, str, str]:
    if prob < 0.30:
        return "Low Risk", "#34d399", "Low"
    elif prob <= 0.70:
        return "Medium Risk", "#fbbf24", "Medium"
    return "High Risk", "#f87171", "High"


RECOMMENDATIONS = {
    "Low": [
        "Keep up the good service — customer appears satisfied.",
        "Consider loyalty rewards to reinforce retention.",
        "Monitor for any changes in usage patterns.",
    ],
    "Medium": [
        "Proactively reach out with a satisfaction check-in.",
        "Offer a retention incentive (discount or bonus).",
        "Review recent support interactions for friction points.",
    ],
    "High": [
        "Immediate action required — assign a retention specialist.",
        "Offer a personalized retention package (discount + service upgrade).",
        "Escalate to management for priority follow-up.",
    ],
}


# ───────────────────────── PREDICTION ENGINE ─────────────────────────
def predict_single(customer_df: pd.DataFrame):
    processed = preprocessor.transform(customer_df)
    for col in preprocessor.feature_names:
        if col not in processed.columns:
            processed[col] = 0
    processed = processed[preprocessor.feature_names]
    prob = model.predict_proba(processed)[0][1]
    pred = 1 if prob >= MODEL_THRESHOLD else 0
    return pred, prob


def predict_bulk(df: pd.DataFrame, progress_bar=None, status_text=None) -> pd.DataFrame:
    n = len(df)
    batch_size = max(1, min(1000, n // 10))

    all_probs = np.zeros(n)
    all_preds = np.zeros(n, dtype=int)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end].copy()
        processed = preprocessor.transform(batch)
        for col in preprocessor.feature_names:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[preprocessor.feature_names]
        probs = model.predict_proba(processed)[:, 1]
        all_probs[start:end] = probs
        all_preds[start:end] = (probs >= MODEL_THRESHOLD).astype(int)
        if progress_bar is not None:
            progress_bar.progress(end / n)
        if status_text is not None:
            status_text.text(f"Predicted {end} / {n} customers")

    df_result = df.copy()
    df_result['Churn_Probability'] = all_probs
    df_result['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in all_preds]
    df_result['Risk_Level'] = [get_risk(p)[0] for p in all_probs]
    return df_result


def generate_global_shap(df: pd.DataFrame, top_n: int = 15) -> Optional[pd.DataFrame]:
    if explainer is None:
        return None
    try:
        processed = preprocessor.transform(df)
        for col in preprocessor.feature_names:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[preprocessor.feature_names]
        return explainer.get_feature_importance(processed)
    except Exception:
        return None


# ───────────────────────── CSV TEMPLATE ─────────────────────────
def generate_csv_template() -> pd.DataFrame:
    np.random.seed(42)
    n = 5
    data = {
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52]),
        'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70]),
        'tenure': np.random.randint(0, 73, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.10]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.55, 0.34, 0.11]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.37, 0.52, 0.11]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.55, 0.11]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.55, 0.11]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.59, 0.11]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.40, 0.49, 0.11]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.39, 0.50, 0.11]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.22, 0.23]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.76, 0.24]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n, p=[0.34, 0.23, 0.22, 0.21]),
        'MonthlyCharges': np.random.uniform(18, 120, n).round(2),
        'TotalCharges': np.random.uniform(18, 8684, n).round(2),
        'Churn': np.random.choice(['Yes', 'No'], n, p=[0.27, 0.73]),
    }
    return pd.DataFrame(data)


# ───────────────────────── DATA VALIDATION ─────────────────────────
def validate_uploaded_data(df: pd.DataFrame) -> List[str]:
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
        return errors

    for col, valid_vals in CATEGORICAL_MAPS.items():
        if col in df.columns:
            invalid = df[~df[col].astype(str).isin([str(v) for v in valid_vals])][col]
            if len(invalid) > 0:
                err_vals = invalid.unique()[:5]
                errors.append(f"Column '{col}' has invalid values: {', '.join(str(v) for v in err_vals)}")

    if 'tenure' in df.columns:
        neg = df[df['tenure'].astype(float) < 0]
        if len(neg) > 0:
            errors.append(f"Found {len(neg)} row(s) with negative tenure")

    if 'MonthlyCharges' in df.columns:
        neg = df[df['MonthlyCharges'].astype(float) < 0]
        if len(neg) > 0:
            errors.append(f"Found {len(neg)} row(s) with negative monthly charges")

    if 'TotalCharges' in df.columns:
        neg = df[df['TotalCharges'].astype(float) < 0]
        if len(neg) > 0:
            errors.append(f"Found {len(neg)} row(s) with negative total charges")

    return errors


# ───────────────────────── INSIGHTS ENGINE ─────────────────────────
def generate_insights(df_result: pd.DataFrame) -> List[str]:
    insights = []
    n = len(df_result)
    high = len(df_result[df_result['Risk_Level'] == 'High Risk'])
    medium = len(df_result[df_result['Risk_Level'] == 'Medium Risk'])
    avg_prob = df_result['Churn_Probability'].mean()

    insights.append(f"**{high/n*100:.0f}%** of customers ({high} out of {n}) are at **high churn risk** — immediate retention campaigns recommended.")

    insights.append(f"Average churn probability across all uploaded customers: **{avg_prob:.1%}**.")

    if 'Contract' in df_result.columns:
        contract_risk = df_result.groupby('Contract')['Churn_Probability'].mean().sort_values(ascending=False)
        highest = contract_risk.index[0]
        insights.append(f"**{highest}** customers have the highest average churn probability ({contract_risk.iloc[0]:.1%}). Consider re-evaluating contract terms.")

    if 'InternetService' in df_result.columns:
        isp_risk = df_result.groupby('InternetService')['Churn_Probability'].mean().sort_values(ascending=False)
        highest_isp = isp_risk.index[0]
        insights.append(f"**{highest_isp}** subscribers show elevated churn risk ({isp_risk.iloc[0]:.1%}). Review service quality and pricing.")

    if medium + high > 0:
        at_risk = medium + high
        insights.append(f"**{at_risk}** customers ({at_risk/n*100:.0f}%) need proactive retention outreach. Estimated value at risk is significant — prioritize high-risk segment first.")

    return insights


# ──────────────────────── CLASSIFICATION METRICS ─────────────────────────
def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}
    try:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics['Accuracy'] = None
    try:
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    except Exception:
        metrics['Precision'] = None
    try:
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    except Exception:
        metrics['Recall'] = None
    try:
        metrics['F1 Score'] = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        metrics['F1 Score'] = None
    try:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics['ROC-AUC'] = None
    return metrics


# ───────────────────────── UI HEADER ─────────────────────────
st.markdown('<div class="main-header"><h1>📊 Churn Analytics Platform</h1><p>Predict, analyze, and act on customer churn risk</p></div>', unsafe_allow_html=True)

if model is None or preprocessor is None:
    st.markdown('<div class="error-msg">Model files not found. Run the training pipeline first:<br><code>python train_pipeline.py</code></div>', unsafe_allow_html=True)
    st.stop()

# ───────────────────────── TABS ─────────────────────────
tab1, tab2 = st.tabs(["🔮 Single Customer Prediction", "📦 Bulk CSV Analysis"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER PREDICTION
# ═══════════════════════════════════════════════════════════
with tab1:
    # ─── MODEL PERFORMANCE METRICS (always visible) ───
    if trained_metrics and len(trained_metrics) > 0:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>🎯 Model Performance</h3>", unsafe_allow_html=True)
        st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-bottom:1rem;'>Test-set metrics for the deployed model (Logistic Regression, trained on 5,634 customers)</div>", unsafe_allow_html=True)
        m_cols = st.columns(len(trained_metrics))
        for i, (name, val) in enumerate(trained_metrics.items()):
            with m_cols[i]:
                st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val:.1%}</div><div class="kpi-label">{name}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Customer Information</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("How long with us (months)?", 0, 72, 12, key="s_tenure")
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"], key="s_contract")
        monthly = st.slider("Monthly charge ($)", 18, 120, 65, key="s_monthly")
        total = st.slider("Total charges to date ($)", 0, 8700, 500, key="s_total")
        gender = st.selectbox("Gender", ["Male", "Female"], key="s_gender")
        senior = st.selectbox("Senior citizen?", ["No", "Yes"], key="s_senior")
        partner = st.selectbox("Has a partner?", ["No", "Yes"], key="s_partner")
        dependents = st.selectbox("Has dependents?", ["No", "Yes"], key="s_dependents")

    with col2:
        internet = st.selectbox("Internet service", ["No", "DSL", "Fiber optic"], key="s_internet")
        phone = st.selectbox("Phone service", ["No", "Yes"], key="s_phone")
        multiple = st.selectbox("Multiple lines", ["No", "No phone service", "Yes"], key="s_multiple")
        online_sec = st.selectbox("Online security", ["No", "No internet service", "Yes"], key="s_online_sec")
        online_bk = st.selectbox("Online backup", ["No", "No internet service", "Yes"], key="s_online_bk")
        device_prot = st.selectbox("Device protection", ["No", "No internet service", "Yes"], key="s_device_prot")
        tech_sup = st.selectbox("Tech support", ["No", "No internet service", "Yes"], key="s_tech_sup")
        streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"], key="s_streaming_tv")
        streaming_mov = st.selectbox("Streaming movies", ["No", "No internet service", "Yes"], key="s_streaming_mov")
        paperless = st.selectbox("Paperless billing?", ["No", "Yes"], key="s_paperless")
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], key="s_payment")

    st.markdown("<div style='text-align:center;margin-top:1rem;'>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, key="btn_single")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        df_single = pd.DataFrame([{
            'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'MultipleLines': multiple,
            'InternetService': internet, 'OnlineSecurity': online_sec,
            'OnlineBackup': online_bk, 'DeviceProtection': device_prot,
            'TechSupport': tech_sup, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_mov, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly, 'TotalCharges': total,
        }])

        with st.spinner("Analyzing customer data..."):
            pred, prob = predict_single(df_single)
            time.sleep(0.3)

        tag, color, label = get_risk(prob)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            st.markdown(f"""
            <div class="risk-gauge">
                <div class="percent" style="color:{color};">{prob:.0%}</div>
                <div class="label" style="color:{color};">{tag}</div>
                <div class="sub">Churn probability</div>
            </div>
            """, unsafe_allow_html=True)

            badge = "✅ STAYING" if pred == 0 else "⚠️ LIKELY TO LEAVE"
            badge_color = "#34d399" if pred == 0 else "#f87171"
            st.markdown(f"<div style='text-align:center;margin-top:0.8rem;'><span class='prediction-badge' style='background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}44;'>Verdict: {badge}</span></div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div style='font-size:0.9rem;color:rgba(255,255,255,0.7);margin-bottom:0.6rem;'><strong>Recommended actions:</strong></div>", unsafe_allow_html=True)
            for rec in RECOMMENDATIONS[label]:
                st.markdown(f"<div class='rec-box' style='border-left-color:{color};'><p>➤ {rec}</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # SHAP Explanation
        shap_exp = None
        if explainer is not None:
            try:
                processed = preprocessor.transform(df_single)
                for col in preprocessor.feature_names:
                    if col not in processed.columns:
                        processed[col] = 0
                processed = processed[preprocessor.feature_names]
                shap_exp = explainer.explain_prediction(processed, sample_idx=0)
            except Exception:
                shap_exp = None

        if shap_exp is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3>Why this prediction?</h3>", unsafe_allow_html=True)
            st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-bottom:1rem;'>Factors influencing the result — green pushes <em>away</em> from churn, red pushes <em>toward</em> churn.</div>", unsafe_allow_html=True)

            top = shap_exp['feature_impact'].head(8)
            max_abs = top['abs_shap'].max() if len(top) > 0 else 1

            for _, row in top.iterrows():
                direction = "toward churn ↑" if row['shap_value'] > 0 else "away from churn ↓"
                bar_color = "#f87171" if row['shap_value'] > 0 else "#34d399"
                pct = abs(row['shap_value']) / max_abs
                disp_name = row['feature'].replace('_', ' ').replace('Yes', '').replace('No', '').strip()
                if not disp_name:
                    disp_name = row['feature']

                st.markdown(f"""
                <div class="shap-bar">
                    <div class="fname">{disp_name}</div>
                    <div class="track">
                        <div class="fill" style="width:{max(5, pct*100)}%;background:{bar_color};"></div>
                    </div>
                    <div class="value">{direction}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3>Why this prediction?</h3>", unsafe_allow_html=True)
            st.markdown("<div style='color:rgba(255,255,255,0.4);text-align:center;padding:1rem 0;'>Detailed explanation not available for this prediction.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Snapshot
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Customer Snapshot</h3>", unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Tenure", f"{tenure} mo")
        s2.metric("Contract", contract[:8] + "…" if len(contract) > 8 else contract)
        s3.metric("Monthly", f"${monthly}")
        s4.metric("Internet", internet)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — BULK CSV ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Upload Customer Data</h3>", unsafe_allow_html=True)

    csv_col1, csv_col2 = st.columns([2, 1])
    with csv_col2:
        template_df = generate_csv_template()
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="churn_sample_template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a sample CSV with the exact columns required"
        )

    with csv_col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help=f"File must contain these columns: {', '.join(REQUIRED_COLUMNS)}",
            key="csv_uploader"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        validation_errors = validate_uploaded_data(df_uploaded)

        if validation_errors:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3>⚠️ Data Validation Errors</h3>", unsafe_allow_html=True)
            for err in validation_errors:
                st.markdown(f'<div class="error-msg">{err}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h3>Data Preview — {len(df_uploaded)} customers loaded</h3>", unsafe_allow_html=True)
            st.dataframe(df_uploaded.head(10), use_container_width=True, height=320)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("📊 Run Bulk Prediction", use_container_width=True, key="btn_bulk"):
                prog_bar = st.progress(0, text="Starting prediction...")
                status_text = st.empty()

                df_results = predict_bulk(df_uploaded, prog_bar, status_text)

                status_text.text(f"✅ Completed — {len(df_results)} customers analyzed")
                time.sleep(0.3)
                status_text.empty()
                prog_bar.empty()

                # ─── RESULTS TABLE ───
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"<h3>📋 Prediction Results</h3>", unsafe_allow_html=True)

                display_cols = [c for c in df_results.columns if c != 'Churn_Probability']
                if 'Predicted_Churn' in display_cols and 'Churn_Probability' in df_results.columns:
                    disp_idx = display_cols.index('Predicted_Churn')
                    display_cols.insert(disp_idx, 'Churn_Probability')
                risk_colors = df_results['Risk_Level'].map({'Low Risk': '#34d399', 'Medium Risk': '#fbbf24', 'High Risk': '#f87171'})

                st.dataframe(df_results, use_container_width=True, height=400)
                st.markdown("</div>", unsafe_allow_html=True)

                # ─── DOWNLOAD ───
                st.markdown('<div class="card">', unsafe_allow_html=True)
                out_buffer = io.StringIO()
                df_results.to_csv(out_buffer, index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=out_buffer.getvalue(),
                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

                # ═══════════════════════════════════════
                # EXECUTIVE DASHBOARD
                # ═══════════════════════════════════════
                st.markdown("## 📈 Executive Dashboard")

                n_total = len(df_results)
                n_high = len(df_results[df_results['Risk_Level'] == 'High Risk'])
                n_medium = len(df_results[df_results['Risk_Level'] == 'Medium Risk'])
                n_low = len(df_results[df_results['Risk_Level'] == 'Low Risk'])
                avg_churn = df_results['Churn_Probability'].mean()
                est_rate = (df_results['Predicted_Churn'] == 'Yes').mean()

                k1, k2, k3, k4, k5, k6 = st.columns(6)
                with k1:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{n_total}</div><div class="kpi-label">Total Customers</div></div>', unsafe_allow_html=True)
                with k2:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#f87171;">{n_high}</div><div class="kpi-label">High Risk</div></div>', unsafe_allow_html=True)
                with k3:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#fbbf24;">{n_medium}</div><div class="kpi-label">Medium Risk</div></div>', unsafe_allow_html=True)
                with k4:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#34d399;">{n_low}</div><div class="kpi-label">Low Risk</div></div>', unsafe_allow_html=True)
                with k5:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_churn:.1%}</div><div class="kpi-label">Avg Churn Probability</div></div>', unsafe_allow_html=True)
                with k6:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{est_rate:.1%}</div><div class="kpi-label">Est. Churn Rate</div></div>', unsafe_allow_html=True)

                # ─── MODEL PERFORMANCE METRICS ───
                if trained_metrics and len(trained_metrics) > 0:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<h3>🎯 Model Performance (Test Set)</h3>", unsafe_allow_html=True)
                    st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-bottom:1rem;'>Trained on 5,634 customers &middot; Evaluated on 1,409 held-out records &middot; Best model: Logistic Regression</div>", unsafe_allow_html=True)
                    m_cols = st.columns(len(trained_metrics))
                    for i, (name, val) in enumerate(trained_metrics.items()):
                        with m_cols[i]:
                            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val:.1%}</div><div class="kpi-label">{name}</div></div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ─── CSV GROUND TRUTH METRICS (if available) ───
                if 'Churn' in df_uploaded.columns:
                    y_true_raw = df_uploaded['Churn'].astype(str).str.strip()
                    y_true = y_true_raw.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(-1)
                    if (y_true >= 0).all():
                        y_true_arr = y_true.astype(int).values
                        y_prob_arr = df_results['Churn_Probability'].values
                        y_pred_arr = (df_results['Predicted_Churn'] == 'Yes').astype(int).values
                        csv_metrics = compute_classification_metrics(y_true_arr, y_prob_arr, y_pred_arr)
                        valid_csv = {k: v for k, v in csv_metrics.items() if v is not None}
                        if valid_csv:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("<h3>🎯 Model Performance (Your Data)</h3>", unsafe_allow_html=True)
                            st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-bottom:1rem;'>Evaluated against ground truth labels in your uploaded CSV.</div>", unsafe_allow_html=True)
                            m_cols = st.columns(len(valid_csv))
                            for i, (name, val) in enumerate(valid_csv.items()):
                                with m_cols[i]:
                                    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val:.1%}</div><div class="kpi-label">{name}</div></div>', unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                if PLOTLY_AVAILABLE:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        risk_counts = df_results['Risk_Level'].value_counts().reset_index()
                        risk_counts.columns = ['Risk_Level', 'Count']
                        colors_map = {'Low Risk': '#34d399', 'Medium Risk': '#fbbf24', 'High Risk': '#f87171'}
                        fig_pie = px.pie(
                            risk_counts, values='Count', names='Risk_Level',
                            color='Risk_Level', color_discrete_map=colors_map,
                            title='Risk Distribution',
                            hole=0.5,
                        )
                        fig_pie.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='rgba(255,255,255,0.8)',
                            title_font_size=16,
                            legend_font_size=12,
                            margin=dict(t=50, b=10, l=10, r=10),
                        )
                        fig_pie.update_traces(textinfo='percent+label', textfont_size=12)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with chart_col2:
                        fig_hist = px.histogram(
                            df_results, x='Churn_Probability', nbins=30,
                            title='Churn Probability Distribution',
                            color_discrete_sequence=['#667eea'],
                            labels={'Churn_Probability': 'Churn Probability'},
                        )
                        fig_hist.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='rgba(255,255,255,0.8)',
                            title_font_size=16,
                            xaxis_title_font_size=12,
                            yaxis_title_font_size=12,
                            margin=dict(t=50, b=10, l=10, r=10),
                            bargap=0.05,
                        )
                        fig_hist.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
                        fig_hist.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
                        st.plotly_chart(fig_hist, use_container_width=True)

                    chart_col3, chart_col4 = st.columns(2)

                    with chart_col3:
                        if 'Contract' in df_results.columns:
                            contract_risk = df_results.groupby('Contract')['Risk_Level'].value_counts().unstack(fill_value=0)
                            fig_contract = go.Figure()
                            for level, color in colors_map.items():
                                if level in contract_risk.columns:
                                    fig_contract.add_trace(go.Bar(
                                        name=level, x=contract_risk.index, y=contract_risk[level],
                                        marker_color=color, text=contract_risk[level], textposition='auto',
                                    ))
                            fig_contract.update_layout(
                                barmode='stack', title='Contract Type vs Risk',
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color='rgba(255,255,255,0.8)', title_font_size=16,
                                legend_font_size=11, margin=dict(t=50, b=10, l=10, r=10),
                            )
                            fig_contract.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
                            fig_contract.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
                            st.plotly_chart(fig_contract, use_container_width=True)

                    with chart_col4:
                        if 'InternetService' in df_results.columns:
                            isp_risk = df_results.groupby('InternetService')['Risk_Level'].value_counts().unstack(fill_value=0)
                            fig_isp = go.Figure()
                            for level, color in colors_map.items():
                                if level in isp_risk.columns:
                                    fig_isp.add_trace(go.Bar(
                                        name=level, x=isp_risk.index, y=isp_risk[level],
                                        marker_color=color, text=isp_risk[level], textposition='auto',
                                    ))
                            fig_isp.update_layout(
                                barmode='stack', title='Internet Service vs Risk',
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color='rgba(255,255,255,0.8)', title_font_size=16,
                                legend_font_size=11, margin=dict(t=50, b=10, l=10, r=10),
                            )
                            fig_isp.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
                            fig_isp.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
                            st.plotly_chart(fig_isp, use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # ─── GLOBAL SHAP ───
                if explainer is not None:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<h3>🔍 Global Feature Importance</h3>", unsafe_allow_html=True)
                    st.markdown("<div style='color:rgba(255,255,255,0.5);font-size:0.85rem;margin-bottom:1rem;'>Top factors driving churn across all uploaded customers.</div>", unsafe_allow_html=True)

                    with st.spinner("Calculating global SHAP values..."):
                        shap_importance = generate_global_shap(df_results)

                    if shap_importance is not None and len(shap_importance) > 0:
                        top_shap = shap_importance.head(15).copy()
                        top_shap['direction'] = top_shap['importance']
                        top_shap = top_shap.sort_values('importance')

                        if PLOTLY_AVAILABLE:
                            fig_shap = px.bar(
                                top_shap, y='feature', x='importance', orientation='h',
                                title=None,
                                color='importance', color_continuous_scale=['#34d399', '#fbbf24', '#f87171'],
                                labels={'importance': 'Mean |SHAP Value|', 'feature': ''},
                            )
                            fig_shap.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color='rgba(255,255,255,0.8)',
                                xaxis_title_font_size=12, yaxis_title_font_size=12,
                                margin=dict(t=10, b=10, l=10, r=10),
                                height=450,
                                coloraxis_showscale=False,
                            )
                            fig_shap.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
                            fig_shap.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
                            st.plotly_chart(fig_shap, use_container_width=True)
                        else:
                            st.dataframe(top_shap, use_container_width=True)
                    else:
                        st.markdown("<div style='color:rgba(255,255,255,0.4);text-align:center;padding:1rem;'>Global SHAP explanation not available for this dataset.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ─── BUSINESS INSIGHTS ───
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<h3>💡 Business Insights</h3>", unsafe_allow_html=True)

                insights = generate_insights(df_results)
                for insight in insights:
                    st.markdown(f'<div class="insight-card"><p>{insight}</p></div>', unsafe_allow_html=True)

                st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
                st.markdown("<h4>Suggested Retention Actions</h4>", unsafe_allow_html=True)
                actions = [
                    "Launch targeted email/SMS campaigns for high-risk customers with personalized discount offers.",
                    "Reach out to month-to-month contract customers with upgrade incentives to switch to annual plans.",
                    "Review internet service quality for Fiber optic customers experiencing issues.",
                    "Create a loyalty program for customers with low churn risk to maintain satisfaction.",
                    "Train support team to identify and flag at-risk customers during interactions."
                ]
                for action in actions:
                    st.markdown(f'<div class="rec-box" style="border-left-color:#667eea;"><p>➤ {action}</p></div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div style='color:rgba(255,255,255,0.4);text-align:center;padding:2rem 0;'>Upload a CSV file or download the sample template to get started.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────── FOOTER ─────────────────────────
st.markdown('<div class="footer">Churn Analytics Platform &middot; Powered by machine learning &middot; Predictions are estimates, not guarantees</div>', unsafe_allow_html=True)
