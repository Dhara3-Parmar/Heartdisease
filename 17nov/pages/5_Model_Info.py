"""Page 5: Model Info - model metadata, feature validation bounds, cache statistics, and limitations."""
import pandas as pd
import streamlit as st

import common as cm
from confidence import FEATURES

cm.page_setup("Model Info", "Model architecture, dataset bounds, cache metrics, and clinical limitations")
model, rules, cache, model_version, data_df, background = cm.load()

st.markdown('<div class="section"><h3>Model Overview</h3></div>', unsafe_allow_html=True)
st.write(f"File: {model_version}")
st.write(f"Type: {type(model).__name__} | Trees: {len(model.estimators_)} | Features: {len(FEATURES)}")
st.write(f"Training data: {len(data_df)} unique patients (duplicates removed)")
counts = data_df["target"].value_counts().sort_index()
st.caption("Target distribution: " + ", ".join(f"{k} = {v}" for k, v in counts.items()))

st.markdown('<div class="section"><h3>Input Feature Ranges (Derived from Dataset)</h3></div>', unsafe_allow_html=True)
table = pd.DataFrame({f: {"kind": rules[f]["kind"], "min": rules[f]["min"], "max": rules[f]["max"], "typical (fill)": rules[f]["fill"]}
                      for f in FEATURES}).T
st.dataframe(table)

st.markdown('<div class="section"><h3>Prediction Cache Statistics</h3></div>', unsafe_allow_html=True)
stats = cache.stats()
st.write(f"Hits: {stats['hits']} | Misses: {stats['misses']} | Coalesced: {stats['waits']} | Hit rate: {stats['hit_rate_%']}% | Size: {stats['size']}")

st.markdown('<div class="section"><h3>Known Limitations</h3></div>', unsafe_allow_html=True)
st.markdown("""
- Trained on a compact cohort (~302 patients); intended for educational exploration rather than clinical diagnosis.
- SHAP values and natural-language explanations reflect model associations, not medical causation.
- Categorical encodings (`sex`, `cp`, `thal`, etc.) follow standard UCI dataset conventions.
""")
cm.footer()
