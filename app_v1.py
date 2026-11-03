
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from input import FEATURES, load_rules, validate_input, predict_batch     
from cache import PredictionCache                                        
from confidence import (CSS, CLASS_NAMES, COLORS, analyse, decide, sweep, 
                        sweep_values, bar_chart, column_chart)

st.set_page_config(
    page_title="HeartCare",
    layout="centered"
)

MODEL_FILE = "heart_model_v20261013_160903.joblib"
DATA_FILE = "heart.csv"
MAX_BATCH_ROWS = 100000


@st.cache_resource
def load_everything():
    model = joblib.load(MODEL_FILE)
    rules = load_rules(DATA_FILE)
    cache = PredictionCache(max_size=512, ttl=600)
    return model, rules, cache, MODEL_FILE


try:
    model, rules, cache, model_version = load_everything()
except FileNotFoundError as e:
    st.error(f"File nahi mili: {e.filename}. app.py ke same folder me rakho.")
    st.stop()


def hint(f):
    r = rules[f]
    if r["kind"] == "category":
        return "/".join(str(int(a)) for a in r["allowed"])
    return f"{r['min']:g} to {r['max']:g}"


def choice(label, f):
    return st.selectbox(f"{label} ({hint(f)})", [None] + rules[f]["allowed"],
                        index=1, format_func=lambda v: "Not known" if v is None else str(int(v)))


def number(label, f, default):
    return st.text_input(f"{label} ({hint(f)})", value=default)


def previous(key, current):
    """Chart ki pichli values (animation start). Session me yaad rakhte hain."""
    old = st.session_state.get("prev", {}).get(key)
    st.session_state.setdefault("prev", {})[key] = list(current)
    return old


st.markdown("""
<style>
.stApp {
    background: #f3f6fb;
}

header, footer {
    visibility: hidden;
}

.main .block-container {
    max-width: 850px;
    padding: 25px 20px;
}

.header {
    background: #142b53;
    color: white;
    padding: 25px;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 25px;
}

.section {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 18px;
    border: 1px solid #e2e8f0;
}

.section h3 {
    color: #142b53;
}

.stButton button {
    background: #142b53;
    color: white;
    width: 100%;
    height: 45px;
    border-radius: 8px;
}

.stButton button:hover {
    background: #244577;
    color: white;
}

@media(max-width: 600px) {
    .main .block-container {
        padding: 12px 10px;
    }

    .header {
        padding: 18px;
    }

    .header h1 {
        font-size: 24px;
    }

    .section {
        padding: 14px;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>HeartCare</h1>
    <p>Heart Disease Risk Prediction System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section">
    <h3>Patient Information</h3>
    <p>Enter patient medical details below.</p>
</div>
""", unsafe_allow_html=True)

st.caption("Koi field khali chhodo ya 'Not known' chuno to median/mode se bhar diya jayega (max 3 fields).")

age = number("Age", "age", "45")
sex = choice("Sex", "sex")
cp = choice("Chest Pain Type", "cp")
trestbps = number("Resting Blood Pressure", "trestbps", "120")
chol = number("Cholesterol", "chol", "200")
fbs = choice("Fasting Blood Sugar", "fbs")
restecg = choice("Resting ECG", "restecg")
thalach = number("Maximum Heart Rate", "thalach", "150")
exang = choice("Exercise Induced Angina", "exang")
oldpeak = number("Oldpeak", "oldpeak", "1.0")
slope = choice("Slope", "slope")
ca = choice("Major Vessels", "ca")
thal = choice("Thal", "thal")

live = st.checkbox("Live update (input badalte hi prediction update ho)")
clicked = st.button("Predict Heart Disease Risk")

if clicked or live:

    record = dict(zip(FEATURES, [
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]))

    check = validate_input(record, rules)

    for message in check["errors"]:
        st.error(message)

    if check["ok"]:
        st.session_state["check"] = check      
    else:
        st.session_state.pop("check", None)

check = st.session_state.get("check")

if check:

    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown("""
<div class="section">
    <h3>Prediction & Confidence</h3>
</div>
""", unsafe_allow_html=True)

    threshold = st.slider("Decision threshold (%): Higher Risk tab maana jaye jab probability isse zyada ho", 1, 99, 50) / 100
    explore = st.selectbox("What-if: ek feature badalke confidence dekho", ["(none)"] + FEATURES)

    start = time.perf_counter()
    key = (model_version, tuple(float(v) for v in check["x"]))
    info = cache.get_or_compute(key, lambda: analyse(model, check["x"]))
    elapsed = (time.perf_counter() - start) * 1000

    for message in check["warnings"]:
        st.warning(message)

    probs = info["probs"]
    prediction = decide(probs, threshold)

    if prediction == 1:
        st.error(f"{CLASS_NAMES[1]} Prediction")
    else:
        st.success(f"{CLASS_NAMES[0]} Prediction")

    st.write("Model Confidence:", round(max(probs) * 100, 2), "%")

    if threshold != 0.5:
        st.caption(f"Custom threshold {threshold * 100:.0f}%: {CLASS_NAMES[1]} probability {probs[1] * 100:.1f}% hai, "
                   f"isliye prediction {CLASS_NAMES[prediction]} dikh raha hai (model ka default threshold 50% hai).")
    st.caption(f"{info['votes'][1]} of {info['n_trees']} trees ne {CLASS_NAMES[1]} chuna | trees ke beech spread (std): {info['std']:.2f}")

    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    st.markdown(bar_chart("Class probabilities (trees ka average)", labels, probs,
                          prev=previous("probs", probs), colors=[COLORS[0], COLORS[1]], threshold=threshold,
                          note=f"Kali line = threshold ({threshold * 100:.0f}%)"), unsafe_allow_html=True)

    votes = info["votes"] / info["n_trees"]
    st.markdown(bar_chart("Tree votes (kitni trees ne kaunsi class chuni)", labels, votes,
                          prev=previous("votes", votes), colors=[COLORS[0], COLORS[1]]), unsafe_allow_html=True)

    hist = info["hist"]
    bins = [f"{i * 10}-{i * 10 + 10}" for i in range(len(hist))]
    st.markdown(column_chart(f"Har tree ka {CLASS_NAMES[1]} probability (%) - kitni trees kis range me",
                             bins, hist / hist.max(), prev=previous("hist", hist / hist.max()),
                             color="#244577", counts=[int(h) for h in hist],
                             note="Bar ke upar tree ginti. Sab trees ek jaisi ho to bars sirf kinaron par aate hain."),
                unsafe_allow_html=True)

    if explore != "(none)":
        values = sweep_values(rules[explore])
        p_values = sweep(model, check["x"], explore, values)
        st.markdown(column_chart(f"What-if: {explore} badalne par {CLASS_NAMES[1]} probability",
                                 [f"{v:g}" for v in values], p_values, prev=previous("sweep_" + explore, p_values),
                                 color="#d64545", note=f"Baaki saari values wahi rakhi gayi hain; sirf {explore} badla."),
                    unsafe_allow_html=True)

    if not live:
        st.caption("Ye result pichhle Predict click ke inputs ka hai. Naye inputs ke liye dobara Predict dabao ya Live update on karo.")

    stats = cache.stats()
    st.caption(f"Is request me {elapsed:.2f} ms | Cache: {stats['hits']} hits, {stats['misses']} misses, "
               f"{stats['waits']} coalesced, hit rate {stats['hit_rate_%']}% (size {stats['size']})")

st.markdown("""
<div class="section">
    <h3>Batch Prediction (CSV)</h3>
    <p>Bahut patients ek saath: CSV me columns age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak, slope, ca, thal hone chahiye.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

data = None

if uploaded is not None:

    try:
        data = pd.read_csv(uploaded)
    except Exception:
        st.error("CSV padh nahi paye. File sahi CSV format me honi chahiye.")

if data is not None:

    clean_names = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    missing_cols = [f for f in FEATURES if f not in clean_names]

    if len(data) == 0:
        st.error("CSV khali hai.")
    elif len(data) > MAX_BATCH_ROWS:
        st.error(f"Max {MAX_BATCH_ROWS} rows allowed hain.")
    else:
        if missing_cols:
            st.warning(f"Ye columns CSV me nahi mile (missing maane gaye): {missing_cols}")

        start = time.perf_counter()
        results = predict_batch(model, rules, data)
        elapsed = (time.perf_counter() - start) * 1000

        rejected = int((results["status"] == "rejected").sum())
        st.write(f"Rows: {len(data)} | Predicted: {len(data) - rejected} | Rejected: {rejected}")
        st.caption(f"{elapsed:.0f} ms ({elapsed / len(data):.3f} ms per row)")

        output = pd.concat([data, results], axis=1)
        st.dataframe(output.head(100))

        st.download_button("Download results", output.to_csv(index=False), "heart_predictions.csv", "text/csv")

st.markdown("""
<div style="text-align:center;color:#64748b;padding:20px;">
    HeartCare | Educational Purpose Only
    <br>
    Not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
