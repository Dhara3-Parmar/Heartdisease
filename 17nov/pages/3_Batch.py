"""Page 3: Batch - vectorized CSV batch predictions with row-level diagnostic suggestions."""
import time

import pandas as pd
import streamlit as st

import common as cm
import validation as V
from confidence import FEATURES
from input import predict_batch

MAX_BATCH_ROWS = 100000
MAX_SUGGEST_ROWS = 200

cm.page_setup("Batch", "Predict multiple patients simultaneously via CSV upload")
model, rules, cache, model_version, data_df, background = cm.load()

st.markdown("""
<div class="section">
    <h3>CSV Batch Upload</h3>
    <p>Required columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
    Invalid rows are flagged with actionable fix suggestions.</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
data = None
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not parse CSV file. Please upload a valid CSV.")

if data is not None:
    clean_names = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    missing_cols = [f for f in FEATURES if f not in clean_names]
    if len(data) == 0:
        st.error("Uploaded CSV is empty.")
    elif len(data) > MAX_BATCH_ROWS:
        st.error(f"Maximum {MAX_BATCH_ROWS} rows allowed.")
    else:
        if missing_cols:
            close = {c: V.match_field(c)[1][:1] for c in clean_names if c not in FEATURES and V.match_field(c)[0] is None}
            hints = "; ".join(f"'{c}' -> '{s[0]}'?" for c, s in close.items() if s)
            st.warning(f"Missing columns in CSV (treated as missing): {missing_cols}." + (f" Possible column typos: {hints}" if hints else ""))

        start = time.perf_counter()
        results = predict_batch(model, rules, data)
        elapsed = (time.perf_counter() - start) * 1000
        rejected_idx = list(results.index[results["status"] == "rejected"])

        suggestion = pd.Series("", index=results.index, name="suggestion")
        renamed = data.copy(); renamed.columns = clean_names
        for i in rejected_idx[:MAX_SUGGEST_ROWS]:
            d = V.diagnose({f: renamed.iloc[i].get(f) for f in FEATURES}, rules, model)
            suggestion.iloc[i] = " | ".join(x["message"] for x in d["issues"] if x["level"] == "error")
        results = pd.concat([results, suggestion], axis=1)

        st.write(f"Rows: {len(data)} | Predicted: {len(data) - len(rejected_idx)} | Rejected: {len(rejected_idx)}")
        st.caption(f"{elapsed:.0f} ms ({elapsed / len(data):.3f} ms per row)")
        if len(rejected_idx) > MAX_SUGGEST_ROWS:
            st.caption(f"Suggestions generated for the first {MAX_SUGGEST_ROWS} rejected rows.")
        output = pd.concat([data, results], axis=1)
        st.dataframe(output.head(100))
        st.download_button("Download results", output.to_csv(index=False), "heart_predictions.csv", "text/csv")

cm.footer()
