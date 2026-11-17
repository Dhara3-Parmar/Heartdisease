"""Page 7: Missing Value Imputation - KNN, MICE & Predictive Modeling (Today Task 3)."""
import pandas as pd
import streamlit as st

import common as cm
import imputation as IMP

cm.page_setup("Missing Value Imputation", "Statistical comparison of KNN, MICE, and Predictive Modeling")
model, rules, cache, model_version, data_df, background = cm.load()

raw_df = pd.read_csv(cm.DATA_FILE)

st.markdown('<div class="section"><h3>1. Dataset Missingness Audit (Explicit & Hidden Codes)</h3>'
            '<p>While heart.csv has zero explicit NaNs, ca=4 (5 rows) and thal=0 (2 rows) represent undocumented missing value codes from the original UCI Cleveland dataset.</p></div>',
            unsafe_allow_html=True)
audit_df = IMP.detect_hidden_missing(raw_df)
st.dataframe(audit_df)

st.markdown('<div class="section"><h3>2. Statistical Comparison: KNN vs MICE vs Predictive Modeling (RF)</h3>'
            '<p>Evaluated under 15% simulated missingness using RMSE, MAE, Accuracy, Macro-F1, and Kolmogorov-Smirnov (KS) distribution preservation tests.</p></div>',
            unsafe_allow_html=True)
eval_df, selection_df, final_df = IMP.evaluate_imputation_methods(raw_df, missing_rate=0.15, seed=42)
st.dataframe(eval_df)

st.markdown('<div class="section"><h3>3. Selected Best Imputation Method per Column</h3></div>', unsafe_allow_html=True)
st.dataframe(selection_df)

st.download_button("Download Imputed Dataset (heart_imputed.csv)", final_df.to_csv(index=False), "heart_imputed.csv", "text/csv")
cm.footer()
