"""
ChurnGuard – Bank Customer Intelligence Platform
Multi-model scoring + Natural Language Querying (real LLM via Groq + rule-based fallback)
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard | Bank Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── Professional Dark Theme ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background-color: #0B1220; }
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1200px;
}
.app-header {
    background: linear-gradient(125deg, #0A1628 0%, #0F2744 45%, #1A4A7A 100%);
    color: #fff;
    padding: 1.4rem 1.6rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}
.app-header h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: -0.02em;
}
.app-header p {
    margin: 0.3rem 0 0 0;
    font-size: 0.9rem;
    color: rgba(219, 234, 254, 0.9) !important;
}
.section-title {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #E0F2FE !important;
    margin: 1.3rem 0 0.7rem 0 !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 2px solid #38BDF8 !important;
    letter-spacing: 0.01em;
}
.metric-card {
    background: #0F1C2E;
    border-radius: 12px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    border: 1px solid #1E3A5F;
    text-align: center;
    height: 100%;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
    margin-bottom: 0.35rem;
}
.metric-card .value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #F0F9FF;
}
.risk-high {
    background: linear-gradient(135deg, #1E3A8A, #1E40AF);
    color: #DBEAFE;
    border: 1px solid #60A5FA;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
}
.risk-medium {
    background: linear-gradient(135deg, #0C4A6E, #0369A1);
    color: #E0F2FE;
    border: 1px solid #38BDF8;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #064E3B, #065F46);
    color: #D1FAE5;
    border: 1px solid #34D399;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
}
.result-panel {
    background: #0F1C2E;
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
    border: 1px solid #1E3A5F;
    margin-top: 1.2rem;
}
.answer-box {
    background: #0F1C2E;
    border-left: 4px solid #38BDF8;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    margin-top: 0.9rem;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #E2E8F0;
}
.flow-box {
    background: #0F2744;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 0.95rem 1.2rem;
    font-size: 0.88rem;
    color: #BAE6FD;
    margin-bottom: 1.2rem;
    line-height: 1.55;
}
.stButton > button {
    border-radius: 10px !important;
    border: 1px solid #1E3A5F !important;
    background: #0F1C2E !important;
    color: #E0F2FE !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: #38BDF8 !important;
    color: #FFFFFF !important;
    background: #132337 !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #0F2744 0%, #1A4A7A 50%, #2563EB 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0A1628 0%, #0F2744 50%, #1A4A7A 100%) !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060D18 0%, #0A1628 100%);
}
section[data-testid="stSidebar"] * { color: #DBEAFE !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(56, 189, 248, 0.2); }
section[data-testid="stSidebar"] .stSelectbox label { color: #7DD3FC !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #0F1C2E !important;
    border-color: #1E3A5F !important;
}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border: 1px solid #1E3A5F !important;
    border-radius: 10px !important;
    background-color: #0F1C2E !important;
    color: #F1F5F9 !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}
[data-baseweb="select"] > div {
    background-color: #0F1C2E !important;
    border-color: #1E3A5F !important;
    border-radius: 10px !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1A4A7A, #38BDF8) !important;
}
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stAlert"] {
    border-radius: 10px;
    background-color: #0F1C2E;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.35rem 0;
}
</style>
""", unsafe_allow_html=True)
# ── Paths & constants ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bank-1.csv"
RANDOM_STATE = 42
FEATURE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "BalanceSalaryRatio", "HasZeroBalance", "HighProductCount",
]
NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "BalanceSalaryRatio", "HasZeroBalance", "HighProductCount",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
PROFILE_PRESETS = {
    "high": {
        "credit_score": 550, "age": 48, "tenure": 2, "balance": 125000.0,
        "num_products": 3, "has_cr_card": 1, "is_active": 0,
        "estimated_salary": 90000.0, "geography": "Germany", "gender": "Female",
    },
    "medium": {
        "credit_score": 620, "age": 42, "tenure": 4, "balance": 80000.0,
        "num_products": 1, "has_cr_card": 1, "is_active": 0,
        "estimated_salary": 95000.0, "geography": "Spain", "gender": "Male",
    },
    "low": {
        "credit_score": 720, "age": 32, "tenure": 6, "balance": 0.0,
        "num_products": 2, "has_cr_card": 1, "is_active": 1,
        "estimated_salary": 110000.0, "geography": "France", "gender": "Male",
    },
    "default": {
        "credit_score": 600, "age": 40, "tenure": 3, "balance": 50000.0,
        "num_products": 1, "has_cr_card": 1, "is_active": 1,
        "estimated_salary": 100000.0, "geography": "France", "gender": "Female",
    },
}
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["HasZeroBalance"] = (out["Balance"] == 0).astype(int)
    out["HighProductCount"] = (out["NumOfProducts"] >= 3).astype(int)
    return out
def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ])
def best_threshold(y_true, probs):
    best_t, best_rec = 0.5, 0.0
    for t in np.arange(0.25, 0.75, 0.05):
        rec = recall_score(y_true, (probs >= t).astype(int), zero_division=0)
        if rec > best_rec:
            best_rec, best_t = rec, float(t)
    return best_t
@st.cache_resource(show_spinner=False)
def load_all():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place bank-1.csv under data/."
        )
    df = pd.read_csv(DATA_PATH)
    df_fe = engineer_features(df)
    X = df_fe[FEATURE_COLS]
    y = df_fe["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", max_depth=12,
            min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
            eval_metric="logloss", n_jobs=-1
        )
    if HAS_LGB:
        candidates["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            scale_pos_weight=scale_pos, random_state=RANDOM_STATE,
            verbose=-1, n_jobs=-1
        )
    models = {}
    rows = []
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        thr = best_threshold(y_test, probs)
        pred = (probs >= thr).astype(int)
        m = {
            "Model": name,
            "Threshold": thr,
            "Recall": float(recall_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probs)),
            "Accuracy": float(accuracy_score(y_test, pred)),
        }
        rows.append(m)
        models[name] = {"pipe": pipe, "threshold": thr, "metrics": m}
    metrics_df = pd.DataFrame(rows).sort_values("Recall", ascending=False)
    default_name = str(metrics_df.iloc[0]["Model"])
    full_probs = models[default_name]["pipe"].predict_proba(X)[:, 1]
    df_fe = df_fe.copy()
    df_fe["ChurnProbability"] = full_probs
    df_fe["RiskLevel"] = pd.cut(
        full_probs,
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return df, df_fe, models, metrics_df, default_name
with st.spinner("Loading models…"):
    df_raw, df_risk, MODELS, METRICS_DF, DEFAULT_MODEL = load_all()
# ---------------------------------------------------------------------------
# Natural-language querying layer (real LLM via Groq + transparent fallback)
# ---------------------------------------------------------------------------
class ChurnDataContext:
    def __init__(self, data):
        self.df = data
    def summary_block(self):
        df = self.df
        geo = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
        act = df.groupby("IsActiveMember")["Exited"].mean()
        prod = df.groupby("NumOfProducts")["Exited"].mean().sort_values(ascending=False)
        thr = df["EstimatedSalary"].quantile(0.75)
        high_income = df[df["EstimatedSalary"] >= thr]["Exited"].mean()
        low_income = df[df["EstimatedSalary"] < thr]["Exited"].mean()
        overall = df["Exited"].mean()
        return {
            "overall_churn_rate_pct": round(overall * 100, 2),
            "churn_by_geography_pct": {k: round(v * 100, 2) for k, v in geo.items()},
            "churn_active_pct": round(act.get(1, float("nan")) * 100, 2),
            "churn_inactive_pct": round(act.get(0, float("nan")) * 100, 2),
            "churn_by_num_products_pct": {int(k): round(v * 100, 2) for k, v in prod.items()},
            "high_income_threshold_salary": round(float(thr), 2),
            "high_income_churn_pct": round(high_income * 100, 2),
            "low_income_churn_pct": round(low_income * 100, 2),
        }
class ChurnAnalyticsEngine:
    def __init__(self, data, model_name="openai/gpt-oss-20b"):
        self.context = ChurnDataContext(data)
        self.model_name = model_name
        # Priority: Streamlit secrets → Environment variable → Hardcoded fallback
        self.api_key = None
        try:
            self.api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
        if not self.api_key:
            self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            self.api_key = "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"
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                self.client = None
    def _ask_llm(self, question):
        stats = self.context.summary_block()
        system_prompt = (
            "You are a churn-analytics assistant for a bank. Answer the "
            "business user's question using ONLY the JSON statistics provided. "
            "Do not invent numbers that are not in the JSON. Be concise (2-4 sentences)."
        )
        user_prompt = f"Statistics (JSON): {stats}\n\nQuestion: {question}"
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=700,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("LLM returned an empty response")
        return f"[LLM-generated answer]\n{text}"
    def _fallback_router(self, question):
        q = question.lower()
        s = self.context.summary_block()
        if re.search(r"high.?income|why.*(leaving|churn)", q):
            answer = (
                f"High-income (salary>={s['high_income_threshold_salary']:,.0f}) churn: "
                f"{s['high_income_churn_pct']}% vs lower-income {s['low_income_churn_pct']}%."
            )
        elif re.search(r"highest|segment", q):
            top_geo = max(s["churn_by_geography_pct"], key=s["churn_by_geography_pct"].get)
            top_prod = max(s["churn_by_num_products_pct"], key=s["churn_by_num_products_pct"].get)
            answer = (
                f"Highest-churn segments -> Geography: {top_geo} "
                f"({s['churn_by_geography_pct'][top_geo]}%), "
                f"Inactive members: {s['churn_inactive_pct']}% vs Active: {s['churn_active_pct']}%, "
                f"NumOfProducts={top_prod} ({s['churn_by_num_products_pct'][top_prod]}%)."
            )
        elif re.search(r"geograph|region", q):
            lines = [f" - {k}: {v}%" for k, v in s["churn_by_geography_pct"].items()]
            answer = "Churn by geography:\n" + "\n".join(lines)
        elif re.search(r"overall|churn rate", q):
            answer = f"Overall churn rate: {s['overall_churn_rate_pct']}%."
        else:
            answer = "Ask about segments, high-income, geography, or overall rate."
        return "[FALLBACK - rule-based router, NOT an LLM response]\n" + answer
    def query(self, question):
        if self.client is not None:
            try:
                return self._ask_llm(question)
            except Exception as e:
                return (
                    f"[LLM call failed: {e}. Falling back to rule-based router.]\n"
                    + self._fallback_router(question)
                )
        return self._fallback_router(question)
engine = ChurnAnalyticsEngine(df_risk)
# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ChurnGuard")
    st.caption("Bank Customer Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Predict Churn", "Ask the Data", "Risk Segments", "Compare Models"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Active scoring model**")
    model_names = list(MODELS.keys())
    ordered = [DEFAULT_MODEL] + [m for m in model_names if m != DEFAULT_MODEL]
    selected_model = st.selectbox(
        "Model", ordered, index=0, label_visibility="collapsed"
    )
    sel = MODELS[selected_model]
    st.markdown(
        f"""
| | |
|---|---|
| Threshold | `{sel['threshold']:.2f}` |
| Recall | `{sel['metrics']['Recall']:.3f}` |
| Precision | `{sel['metrics']['Precision']:.3f}` |
| ROC-AUC | `{sel['metrics']['ROC-AUC']:.3f}` |
"""
    )
    st.caption("Primary metric: Recall · Class-balanced")
    st.markdown("---")
    st.info(f"LLM mode active: **{engine.client is not None}**")
# ── Page: Predict ────────────────────────────────────────────────────────────
if page == "Predict Churn":
    st.markdown(
        f"""
        <div class="app-header">
            <h1>Customer Churn Prediction</h1>
            <p>Score a customer with <b>{selected_model}</b> · threshold optimised for Recall.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if "form" not in st.session_state:
        st.session_state.form = PROFILE_PRESETS["default"].copy()
    st.markdown('<div class="section-title">Quick-fill example profiles</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        if st.button("High-risk example", use_container_width=True):
            st.session_state.form = PROFILE_PRESETS["high"].copy()
            st.rerun()
    with e2:
        if st.button("Medium-risk example", use_container_width=True):
            st.session_state.form = PROFILE_PRESETS["medium"].copy()
            st.rerun()
    with e3:
        if st.button("Low-risk example", use_container_width=True):
            st.session_state.form = PROFILE_PRESETS["low"].copy()
            st.rerun()
    d = st.session_state.form
    st.markdown('<div class="section-title">Customer details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 300, 850, int(d["credit_score"]))
        age = st.slider("Age", 18, 92, int(d["age"]))
        tenure = st.slider("Tenure (years)", 0, 10, int(d["tenure"]))
        balance = st.number_input(
            "Balance (€)", min_value=0.0, value=float(d["balance"]), step=1000.0
        )
    with c2:
        num_products = st.selectbox(
            "Number of Products", [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(int(d["num_products"])),
        )
        has_cr_card = st.selectbox(
            "Has Credit Card", [0, 1], index=int(d["has_cr_card"]),
            format_func=lambda x: "Yes" if x else "No",
        )
        is_active = st.selectbox(
            "Is Active Member", [0, 1], index=int(d["is_active"]),
            format_func=lambda x: "Yes" if x else "No",
        )
        estimated_salary = st.number_input(
            "Estimated Salary (€)", min_value=0.0,
            value=float(d["estimated_salary"]), step=1000.0,
        )
    with c3:
        geography = st.selectbox(
            "Geography", ["France", "Spain", "Germany"],
            index=["France", "Spain", "Germany"].index(d["geography"]),
        )
        gender = st.selectbox(
            "Gender", ["Female", "Male"],
            index=["Female", "Male"].index(d["gender"]),
        )
        st.markdown("")
        go = st.button("Score Customer", type="primary", use_container_width=True)
    # Keep form state aligned with current widget values (manual edits)
    st.session_state.form = {
        "credit_score": credit_score,
        "age": age,
        "tenure": tenure,
        "balance": float(balance),
        "num_products": int(num_products),
        "has_cr_card": int(has_cr_card),
        "is_active": int(is_active),
        "estimated_salary": float(estimated_salary),
        "geography": geography,
        "gender": gender,
    }
    if go:
        row = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary,
        }
        Xc = engineer_features(pd.DataFrame([row]))[FEATURE_COLS]
        pipe = MODELS[selected_model]["pipe"]
        thr = MODELS[selected_model]["threshold"]
        prob = float(pipe.predict_proba(Xc)[0, 1])
        will_churn = prob >= thr
        if prob >= 0.60:
            risk_html = '<div class="risk-high">HIGH RISK</div>'
            action = "Contact a retention specialist and prepare a personalised offer."
        elif prob >= 0.30:
            risk_html = '<div class="risk-medium">MEDIUM RISK</div>'
            action = "Add to a nurturing sequence and monitor engagement for 30 days."
        else:
            risk_html = '<div class="risk-low">LOW RISK</div>'
            action = "Standard relationship management — no urgent action required."
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown(f"#### Result · {selected_model}")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f'<div class="metric-card"><div class="label">Churn Probability</div>'
                f'<div class="value">{prob*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with r2:
            label = "Likely to Churn" if will_churn else "Likely to Stay"
            st.markdown(
                f'<div class="metric-card"><div class="label">Prediction</div>'
                f'<div class="value" style="font-size:1.15rem;">{label}</div></div>',
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(risk_html, unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"Decision threshold = {thr:.2f} (Recall-optimised for {selected_model})"
        )
        st.info(f"**Recommended action:** {action}")
        st.markdown("</div>", unsafe_allow_html=True)
# ── Page: Ask the Data ───────────────────────────────────────────────────────
elif page == "Ask the Data":
    st.markdown(
        """
        <div class="app-header">
            <h1>Querying Data Using LLMs</h1>
            <p>Natural language interaction with the dataset — real LLM (Groq) when available, otherwise transparent rule-based fallback.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="flow-box">
        <b>How it works</b><br>
        User question → Groq LLM (if key present) using pre-computed statistics<br>
        → If LLM unavailable / fails → transparent rule-based router<br>
        <i>Every figure is calculated from the dataset. Nothing is invented.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Suggested business questions</div>', unsafe_allow_html=True)
    suggestions = [
        "Which customer segments have the highest churn risk?",
        "Why are high-income customers leaving the bank?",
        "Which geography has the highest churn?",
        "How does being an inactive member affect churn?",
        "What is the overall churn rate?",
        "Show me the risk segment summary",
        "How does number of products relate to churn?",
        "What is the relationship between age and churn?",
        "How does account balance relate to churn?",
        "How does tenure relate to churn?",
        "How does credit score relate to churn?",
        "What is the churn rate by gender?",
    ]
    if "ask_query" not in st.session_state:
        st.session_state.ask_query = ""
    if "ask_answer" not in st.session_state:
        st.session_state.ask_answer = ""
    def _set_suggestion(q: str):
        st.session_state.ask_query = q
        st.session_state.ask_answer = engine.query(q)
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        cols[i % 2].button(
            q,
            key=f"sug_{i}",
            use_container_width=True,
            on_click=_set_suggestion,
            args=(q,),
        )
    st.markdown('<div class="section-title">Or type your own question</div>', unsafe_allow_html=True)
    with st.form("ask_form", clear_on_submit=False):
        typed = st.text_input(
            "Question",
            value=st.session_state.ask_query,
            placeholder="e.g. Which customer segments have the highest churn risk?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Get answer", type="primary", use_container_width=True)
    if submitted and typed.strip():
        st.session_state.ask_query = typed.strip()
        st.session_state.ask_answer = engine.query(typed.strip())
    if st.session_state.ask_answer:
        st.markdown(
            f'<div class="answer-box"><strong>Answer</strong><br><br>'
            f'{st.session_state.ask_answer}</div>',
            unsafe_allow_html=True,
        )
        st.caption("All figures are computed live from the customer dataset.")
# ── Page: Risk Segments ──────────────────────────────────────────────────────
elif page == "Risk Segments":
    st.markdown(
        """
        <div class="app-header">
            <h1>Customer Risk Segmentation</h1>
            <p>Portfolio view by predicted risk — prioritise retention where it matters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    X_all = engineer_features(df_raw)[FEATURE_COLS]
    probs_all = MODELS[selected_model]["pipe"].predict_proba(X_all)[:, 1]
    view = df_raw.copy()
    view["ChurnProbability"] = probs_all
    view["RiskLevel"] = pd.cut(
        probs_all,
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    summary = (
        view.groupby("RiskLevel", observed=True)
        .agg(
            Customers=("Exited", "count"),
            Churned=("Exited", "sum"),
            AvgProb=("ChurnProbability", "mean"),
            AvgAge=("Age", "mean"),
            PctInactive=("IsActiveMember", lambda x: (1 - x.mean()) * 100),
            AvgBalance=("Balance", "mean"),
        )
        .reset_index()
    )
    summary["Churn Rate %"] = (
        summary["Churned"] / summary["Customers"] * 100
    ).round(1)
    summary["Avg Age"] = summary["AvgAge"].round(1)
    summary["Avg Balance (€)"] = summary["AvgBalance"].round(0)
    summary["% Inactive"] = summary["PctInactive"].round(1)
    summary["Avg Probability"] = summary["AvgProb"].round(3)
    st.caption(f"Scored with **{selected_model}**")
    st.dataframe(
        summary[
            [
                "RiskLevel", "Customers", "Churned", "Churn Rate %",
                "Avg Probability", "Avg Age", "% Inactive", "Avg Balance (€)",
            ]
        ].rename(columns={"RiskLevel": "Risk Level"}),
        use_container_width=True,
        hide_index=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Customers by risk level**")
        st.bar_chart(summary.set_index("RiskLevel")["Customers"], color="#1A4A7A")
    with c2:
        st.markdown("**Actual churn rate by risk level**")
        st.bar_chart(summary.set_index("RiskLevel")["Churn Rate %"], color="#2563EB")
    st.markdown("---")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            '<div class="risk-high">HIGH RISK</div>'
            '<p style="font-size:0.85rem;margin-top:0.5rem;color:#94A3B8;">'
            "Specialist contact + personalised incentive.</p>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="risk-medium">MEDIUM RISK</div>'
            '<p style="font-size:0.85rem;margin-top:0.5rem;color:#94A3B8;">'
            "Nurturing sequence + 30-day monitoring.</p>",
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            '<div class="risk-low">LOW RISK</div>'
            '<p style="font-size:0.85rem;margin-top:0.5rem;color:#94A3B8;">'
            "Standard relationship management.</p>",
            unsafe_allow_html=True,
        )
# ── Page: Compare Models ─────────────────────────────────────────────────────
else:
    st.markdown(
        """
        <div class="app-header">
            <h1>Model Comparison</h1>
            <p>All models trained with class balancing · ranked by Recall (primary metric).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show = METRICS_DF.copy()
    for col in ["Threshold", "Recall", "Precision", "F1", "ROC-AUC", "Accuracy"]:
        show[col] = show[col].map(lambda x: f"{x:.3f}")
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-title">Performance comparison</div>', unsafe_allow_html=True)
    chart_df = METRICS_DF.set_index("Model")[["Recall", "Precision", "F1", "ROC-AUC"]]
    st.bar_chart(chart_df)
    st.markdown(
        """
**Design choices**
- **Primary metric:** Recall — minimise customers who churn but are predicted to stay
- **Class imbalance:** `class_weight='balanced'` / `scale_pos_weight` (no SMOTE)
- **Threshold:** swept per model to maximise Recall on the hold-out set
- **Split:** stratified 80/20 so the ~20% churn rate is preserved
"""
    )