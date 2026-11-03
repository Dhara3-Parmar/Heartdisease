import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from input import FEATURES, load_rules, validate_input, predict_batch     
from cache import PredictionCache                                         
from confidence import (CSS, CLASS_NAMES, COLORS, analyse, decide, sweep, 
                        sweep_values, bar_chart, column_chart)
import components as ui                                                    
import shap_values as sv                                                  
from feedback import add_feedback                                        

st.set_page_config(
    page_title="HeartCare",
    layout="centered"
)

MODEL_FILE = "heart_model_v20261013_160903.joblib"
DATA_FILE = "heart.csv"
FEEDBACK_FILE = "feedback.csv"
MAX_BATCH_ROWS = 100000
MAX_UNKNOWN = 3


@st.cache_resource
def load_everything():
    model = joblib.load(MODEL_FILE)
    rules = load_rules(DATA_FILE)
    cache = PredictionCache(max_size=512, ttl=600)
    df = pd.read_csv(DATA_FILE).drop_duplicates()
    background = sv.build_background(df, k=20)
    return model, rules, cache, MODEL_FILE, df, background


try:
    model, rules, cache, model_version, data_df, background = load_everything()
except FileNotFoundError as e:
    st.error(f"File nahi mili: {e.filename}. app.py ke same folder me rakho.")
    st.stop()

SECTIONS = [
    ("Basic", ["age", "sex"]),
    ("Chest & heart rate", ["cp", "exang", "thalach", "oldpeak", "slope"]),
    ("Tests (blood / ECG)", ["trestbps", "chol", "fbs", "restecg"]),
    ("Scan results", ["ca", "thal"]),
]
LABELS = {"age": "Age", "sex": "Sex", "cp": "Chest Pain Type", "trestbps": "Resting Blood Pressure",
          "chol": "Cholesterol", "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
          "thalach": "Maximum Heart Rate", "exang": "Exercise Induced Angina", "oldpeak": "Oldpeak",
          "slope": "Slope", "ca": "Major Vessels", "thal": "Thal"}
HELP = {"age": "Years me.", "sex": "Dataset ka code: 0 ya 1.", "cp": "Chest pain type ka code (0 se 3).",
        "trestbps": "Resting blood pressure, mm Hg.", "chol": "Serum cholesterol, mg/dl.",
        "fbs": "Fasting blood sugar ka code (0 ya 1).", "restecg": "Resting ECG result ka code (0 se 2).",
        "thalach": "Exercise me maximum heart rate, beats per minute.", "exang": "Exercise se chest pain hua? 0 ya 1.",
        "oldpeak": "Exercise ke baad ST depression (number).", "slope": "ST segment slope ka code (0 se 2).",
        "ca": "Major vessels ki ginti (dataset code 0 se 4).", "thal": "Thal test ka code (0 se 3)."}
WHOLE = {"age", "trestbps", "chol", "thalach"}       


def previous(key, current):
    """Chart ki pichli values (animation start). Session me yaad rakhte hain."""
    old = st.session_state.get("prev", {}).get(key)
    st.session_state.setdefault("prev", {})[key] = list(current)
    return old


def default_for(f):
    r = rules[f]
    if r["kind"] == "number":
        return int(round(r["fill"])) if f in WHOLE else float(r["fill"])
    return int(r["fill"])


def reset_inputs():
    for f in FEATURES:
        st.session_state["in_" + f] = default_for(f)
    st.session_state["unknown"] = []
    st.session_state.pop("check", None)


def widget(f):
    """Number -> slider, 2 se 4 options -> radio, zyada options -> selectbox. Default median/mode."""
    r = rules[f]; key = "in_" + f
    if r["kind"] == "number":
        if f in WHOLE:
            st.session_state.setdefault(key, default_for(f))
            v = st.slider(LABELS[f], int(r["min"]), int(r["max"]), key=key, help=HELP[f])
        else:
            st.session_state.setdefault(key, default_for(f))
            v = st.slider(LABELS[f], float(r["min"]), float(r["max"]), step=0.1, key=key, help=HELP[f])
        st.caption(ui.percentile_text(v, data_df[f].values))
        return v
    options = [int(a) for a in r["allowed"]]
    st.session_state.setdefault(key, default_for(f))
    if len(options) <= 4:
        return st.radio(LABELS[f], options, key=key, horizontal=True, help=HELP[f])
    return st.selectbox(LABELS[f], options, key=key, help=HELP[f])


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


st.caption("Sliders/options se values chuno. Koi value pata na ho to neeche 'Values I don't know' me chuno (max 3), wo median/mode se bhar di jayegi.")

top = st.empty()
st.markdown(CSS + ui.CSS + sv.SHAP_CSS, unsafe_allow_html=True)

values = {}
for title, feats in SECTIONS:
    st.markdown(f'<div class="section"><h3>{title}</h3></div>', unsafe_allow_html=True)
    for f in feats:
        values[f] = widget(f)

unknown = st.multiselect("Values I don't know (max 3)", [LABELS[f] for f in FEATURES], key="unknown")
too_many = len(unknown) > MAX_UNKNOWN
if too_many:
    st.error(f"Maximum {MAX_UNKNOWN} values 'unknown' ho sakti hain. Ek-do hata do.")

live = st.checkbox("Live update (slider hilate hi prediction update ho)", value=True)
clicked = st.button("Predict Heart Disease Risk")
st.button("Reset to typical values", on_click=reset_inputs)

if (clicked or live) and not too_many:

    record = {f: (None if LABELS[f] in unknown else values[f]) for f in FEATURES}
    check = validate_input(record, rules)

    for message in check["errors"]:
        st.error(message)

    if check["ok"]:
        st.session_state["check"] = check       
    else:
        st.session_state.pop("check", None)

if too_many:
    st.session_state.pop("check", None)        

check = st.session_state.get("check")

if check:

    st.markdown("""
<div class="section">
    <h3>Prediction & Confidence</h3>
</div>
""", unsafe_allow_html=True)

    threshold = st.slider("Decision threshold (%): Higher Risk tab maana jaye jab probability isse zyada ho", 1, 99, 50, key="threshold") / 100
    explore = st.selectbox("Risk curve: ek feature badalke dekho", FEATURES, index=0, key="explore")
    top_n = st.slider("SHAP: kitne top features dikhayein", 3, 13, 8, key="top_n")

    start = time.perf_counter()
    key = (model_version, tuple(float(v) for v in check["x"]))
    info = cache.get_or_compute(key, lambda: analyse(model, check["x"]))
    elapsed = (time.perf_counter() - start) * 1000

    for message in check["warnings"]:
        st.warning(message)

    filled = list(check["filled"])
    probs = info["probs"]
    prediction = decide(probs, threshold)

    p1 = float(probs[1])
    prev_card = st.session_state.get("last_p1")
    st.session_state["last_p1"] = p1
    top.markdown(ui.risk_card(p1, threshold, prev_card, filled), unsafe_allow_html=True)

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
                             note="Bar ke upar tree ginti."), unsafe_allow_html=True)

    x = check["x"]; r = rules[explore]; fi = FEATURES.index(explore)
    xs = sweep_values(r, 25) if r["kind"] == "number" else sweep_values(r)
    ps = sweep(model, x, explore, xs)
    st.markdown(ui.curve_chart(f"Risk curve: {explore} badalne par risk", explore, xs, ps, float(x[fi]), p1,
                               data_df[explore].values, r["kind"] == "category", threshold), unsafe_allow_html=True)

    st.markdown("""
<div class="section">
    <h3>Why this prediction? (SHAP values)</h3>
    <p>Har feature ne risk ko average patient se kitna upar/neeche dhakela. Ye Shapley values
    haath se nikali gayi hain (13 features ke saare combinations).</p>
</div>
""", unsafe_allow_html=True)

    skey = (model_version, "shap", tuple(float(v) for v in x), len(background))
    sinfo = cache.get_or_compute(skey, lambda: sv.explain(model, x, background))
    inspect = st.selectbox("Ek feature ko detail me dekho", ["(none)"] + FEATURES, key="inspect")
    sel = None if inspect == "(none)" else inspect

    old_phi = previous("shap", sinfo["phi"])
    shap_prev = dict(zip(FEATURES, old_phi)) if old_phi else None

    st.caption(sv.plain_summary(sinfo, filled))
    st.markdown(sv.shap_chart(sinfo, top_n, sel, prev=shap_prev, filled=filled), unsafe_allow_html=True)
    if sel:
        st.markdown(sv.feature_detail(sinfo, sel, background.mean(axis=0), filled), unsafe_allow_html=True)
    st.caption("Transparency: ye model ke patterns hain, medical karan (cause) nahi. Reference = dataset ke 20 patients ka average. "
               "Base + saare contributions = final probability (exactly).")

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
<div class="section">
    <h3>Feedback</h3>
    <p>Ye app kaisa laga? Isse hum interface sudharte hain.</p>
</div>
""", unsafe_allow_html=True)

fb_device = st.radio("Aap kis device par ho?", ["Phone", "Tablet", "Laptop"], horizontal=True, key="fb_device")
fb_task = st.radio("Kya aap apna kaam (risk dekhna) bina madad ke kar paye?", ["Yes", "Partly", "No"], horizontal=True, key="fb_task")
fb_ease = st.slider("Use karna kitna aasaan laga? (1 = mushkil, 5 = bahut aasaan)", 1, 5, 4, key="fb_ease")
fb_text = st.text_area("Kuch atka ya confusing laga? (optional)", key="fb_text", max_chars=500)
if st.button("Send feedback"):
    try:
        add_feedback(FEEDBACK_FILE, fb_device, fb_task, fb_ease, fb_text)
        st.success("Shukriya! Feedback save ho gaya.")
    except Exception as e:
        st.error(f"Feedback save nahi hua: {e}")

st.markdown("""
<div style="text-align:center;color:#64748b;padding:20px;">
    HeartCare | Educational Purpose Only
    <br>
    Not a substitute for professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
