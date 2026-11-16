"""HeartCare - Home page. All workflow pages reside in the `pages/` folder."""
import streamlit as st

import common as cm

cm.page_setup("HeartCare", "Heart Disease Risk Prediction System")
cm.load()

st.markdown("""
<div class="section">
    <h3>How to Use HeartCare</h3>
    <p>Each workflow is organized into a dedicated page. Select a page from the left sidebar or use the navigation buttons below.</p>
</div>
""", unsafe_allow_html=True)

PAGES = [
    ("pages/1_Predict.py", "1. Predict", "Enter patient details via interactive sliders or quick text entry with automatic validation and instant risk estimation."),
    ("pages/2_Explain.py", "2. Explain", "Understand why the prediction was made: confidence-calibrated natural language summary, exact SHAP chart, and risk curve."),
    ("pages/3_Batch.py", "3. Batch", "Upload a CSV of multiple patients for fast vectorized batch prediction and row-level fix suggestions."),
    ("pages/4_Feedback.py", "4. Feedback", "Submit usability ratings and inspect aggregated feedback statistics."),
    ("pages/5_Model_Info.py", "5. Model Info", "Review model architecture, feature ranges, cache performance, and clinical limitations."),
    ("pages/6_Latency.py", "6. API Latency & Quantization", "Profile request-response times, inspect int16/float32 model quantization, and benchmark caching speedup."),
    ("pages/7_Imputation.py", "7. Missing Value Imputation", "Compare KNN, MICE, and Predictive Modeling (RF) imputation methods and select the optimal method per column."),
    ("pages/8_Documentation.py", "8. Project Documentation", "Comprehensive summary of methodologies, experimental results, deployment challenges, and future recommendations."),
]
for path, label, text in PAGES:
    st.markdown(f'<div class="page-card"><b>{label}</b><br>{text}</div>', unsafe_allow_html=True)
    cm.nav_link(path, f"Open {label}")

if st.session_state.get("check"):
    st.markdown('<div class="notice">Your latest patient prediction is saved in session state. Visit the <b>Explain</b> page to inspect feature attributions.</div>',
                unsafe_allow_html=True)

cm.footer()
