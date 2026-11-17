"""Page 6: API Latency Profiling, Model Quantization & Caching Benchmark (Today Task 1)."""
import streamlit as st

import common as cm
import latency as L
from confidence import bar_chart

cm.page_setup("API Latency & Quantization", "Request-response profiling, model quantization, and caching benchmark")
model, rules, cache, model_version, data_df, background = cm.load()

qmodel = L.QuantizedForest(model)
size_info = qmodel.size_report()

st.markdown('<div class="section"><h3>1. Model Quantization (Int16 / Float32 Flat Tree Arrays)</h3>'
            '<p>All 100 Random Forest trees are packed into contiguous 2D NumPy arrays (int16 child indices, int8 feature indices, float32 thresholds and leaf probabilities).</p></div>',
            unsafe_allow_html=True)
st.write(f"Original Tree Arrays: {size_info['original_tree_kb']} KB | Quantized Tree Arrays: {size_info['quantized_tree_kb']} KB | "
         f"Memory Saved: {size_info['memory_reduction_%']}% ({size_info['compression_ratio']}x smaller)")

st.markdown('<div class="section"><h3>2. Request-Response Latency Benchmark</h3>'
            '<p>End-to-end latency breakdown across input validation, confidence estimation, exact SHAP computation, and NLG explanation generation.</p></div>',
            unsafe_allow_html=True)

summary_df, cstats = L.profile_pipeline(model, qmodel, rules, data_df, n_requests=20)
st.dataframe(summary_df)

max_ms = float(summary_df["Mean Total (ms)"].max()) or 1.0
vals = [float(v) / max_ms for v in summary_df["Mean Total (ms)"]]
labels = list(summary_df["Pipeline Mode"])
st.markdown(bar_chart("Relative End-to-End Latency (shorter bar = faster response)", labels, vals,
                      colors=["#b42323", "#244577", "#177245"],
                      note="Quantized vectorization + PredictionCache drastically reduces end-to-end response time."),
            unsafe_allow_html=True)

cm.footer()
