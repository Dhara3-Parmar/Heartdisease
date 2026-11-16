"""
HeartCare - shared utilities across all Streamlit pages: model/rules/cache loading, page setup, input widgets, and safe navigation.
"""
import joblib
import pandas as pd
import streamlit as st

import theme
import shap_values as sv
import components as ui
import explain_text as ex
from confidence import CSS as CHART_CSS, FEATURES
from input import load_rules
from cache import PredictionCache

MODEL_FILE = "heart_model_v20261013_160903.joblib"
DATA_FILE = "heart.csv"
FEEDBACK_FILE = "feedback.csv"
MAX_UNKNOWN = 3

SECTIONS = [
    ("Basic", ["age", "sex"]),
    ("Chest & Heart Rate", ["cp", "exang", "thalach", "oldpeak", "slope"]),
    ("Blood & ECG Tests", ["trestbps", "chol", "fbs", "restecg"]),
    ("Scan Results", ["ca", "thal"]),
]
LABELS = {"age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting Blood Pressure",
          "chol": "Cholesterol", "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
          "thalach": "Maximum Heart Rate", "exang": "Exercise Induced Angina", "oldpeak": "Oldpeak",
          "slope": "Slope", "ca": "Major Vessels", "thal": "Thal"}
HELP = {"age": "Age in years.", "sex": "Dataset code: 0 or 1.", "cp": "Chest pain type code (0 to 3).",
        "trestbps": "Resting blood pressure in mm Hg.", "chol": "Serum cholesterol in mg/dL.",
        "fbs": "Fasting blood sugar code (0 or 1).", "restecg": "Resting ECG result code (0 to 2).",
        "thalach": "Maximum heart rate achieved during exercise (bpm).", "exang": "Exercise induced angina (0 or 1).",
        "oldpeak": "ST depression induced by exercise relative to rest.", "slope": "Slope of peak exercise ST segment (0 to 2).",
        "ca": "Number of major vessels colored by fluoroscopy (0 to 4).", "thal": "Thalassemia test code (0 to 3)."}
WHOLE = {"age", "trestbps", "chol", "thalach"}
KEEP = ("unknown", "threshold", "top_n", "inspect", "explore", "quick")


@st.cache_resource
def load_everything():
    model = joblib.load(MODEL_FILE)
    rules = load_rules(DATA_FILE)
    cache = PredictionCache(max_size=512, ttl=600)
    df = pd.read_csv(DATA_FILE).drop_duplicates()
    background = sv.build_background(df, k=20)
    return model, rules, cache, MODEL_FILE, df, background


def load():
    """Load shared model, rules, cache, dataset, and SHAP background."""
    try:
        return load_everything()
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}. Please keep all project files in the same folder.")
        st.stop()


def page_setup(title, subtitle):
    """Configure page settings, preserve session widget state, and render theme CSS + header."""
    st.set_page_config(page_title=f"HeartCare - {title}", layout="centered")
    keep_inputs()
    st.markdown(theme.CSS + CHART_CSS + ui.CSS + sv.SHAP_CSS + ex.CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="header"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def keep_inputs():
    """Preserve widget values in st.session_state when navigating across pages."""
    for k in list(st.session_state.keys()):
        if k.startswith("in_") or k in KEEP:
            st.session_state[k] = st.session_state[k]


def nav_link(path, label):
    """Safe cross-version page link: handles KeyError('url_pathname') in newer Streamlit versions by falling back to st.switch_page."""
    try:
        st.page_link(path, label=label)
    except Exception:
        if hasattr(st, "switch_page"):
            if st.button(label, key=f"nav_{path}_{label}"):
                try:
                    st.switch_page(path)
                except Exception:
                    st.info(f"Please select '{label}' from the left sidebar.")
        else:
            st.caption(f"Select '{label}' from the left sidebar.")


def previous(key, current):
    old = st.session_state.get("prev", {}).get(key)
    st.session_state.setdefault("prev", {})[key] = list(current)
    return old


def default_for(rules, f):
    r = rules[f]
    if r["kind"] == "number":
        return int(round(r["fill"])) if f in WHOLE else float(r["fill"])
    return int(r["fill"])


def set_input(rules, f, value):
    r = rules[f]
    if r["kind"] == "number":
        v = float(value)
        lo, hi = r["min"], r["max"]
        v = min(max(v, lo), hi)
        st.session_state["in_" + f] = int(round(v)) if f in WHOLE else round(v, 1)
    else:
        st.session_state["in_" + f] = int(value)


def reset_inputs(rules):
    for f in FEATURES:
        st.session_state["in_" + f] = default_for(rules, f)
    st.session_state["unknown"] = []
    st.session_state.pop("check", None)
    st.session_state.pop("qissues", None)


def widget(rules, data_df, f):
    r = rules[f]; key = "in_" + f
    st.session_state.setdefault(key, default_for(rules, f))
    if r["kind"] == "number":
        if f in WHOLE:
            v = st.slider(LABELS[f], int(r["min"]), int(r["max"]), key=key, help=HELP[f])
        else:
            v = st.slider(LABELS[f], float(r["min"]), float(r["max"]), step=0.1, key=key, help=HELP[f])
        st.caption(ui.percentile_text(v, data_df[f].values))
        return v
    options = [int(a) for a in r["allowed"]]
    if len(options) <= 4:
        return st.radio(LABELS[f], options, key=key, horizontal=True, help=HELP[f])
    return st.selectbox(LABELS[f], options, key=key, help=HELP[f])


def need_check():
    """Ensure a valid patient input exists in session_state before rendering explanation pages."""
    check = st.session_state.get("check")
    if not check:
        st.markdown('<div class="notice">Please enter patient details on the <b>Predict</b> page first to view explanations here.</div>',
                    unsafe_allow_html=True)
        nav_link("pages/1_Predict.py", "Go to Predict Page")
        st.stop()
    return check


def footer():
    st.markdown('<div style="text-align:center;color:#475569;padding:20px;">HeartCare | Educational Purpose Only<br>'
                'Not a substitute for professional medical diagnosis.</div>', unsafe_allow_html=True)
