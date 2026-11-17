"""Page 9: Data Cleaning, Schema Standardization & Outlier Pipeline (Today Tasks 1, 2 & 3)."""
import pandas as pd
import streamlit as st

import common as cm
import cleaning as CL
import schema_parser as SP
import outliers as OL

cm.page_setup("Data Cleaning & Outlier Pipeline", "Duplicate/column removal, dynamic schema standardization, and IQR/Z-Score/DBSCAN outlier treatment")
cm.load()

raw_df = pd.read_csv(cm.DATA_FILE)

st.markdown('<div class="section"><h3>1. Duplicate Row & Irrelevant Column Removal (Today Task 1)</h3>'
            '<p>Automatically detects and logs exact/feature-level duplicate rows and removes constant, ID, high-missingness, or collinear mirror columns while protecting all 13 critical clinical features.</p></div>',
            unsafe_allow_html=True)

messy_sample = CL.create_messy_sample(raw_df)
clean_df, dup_log, col_log, summary = CL.clean_duplicates_and_columns(messy_sample)
st.write(f"Initial Shape: {summary['initial_rows']} rows x {summary['initial_cols']} cols | "
         f"Final Shape: {summary['final_rows']} rows x {summary['final_cols']} cols | "
         f"Duplicates Removed: {summary['duplicates_removed']} | "
         f"Irrelevant Columns Removed: {summary['irrelevant_cols_removed']} ({', '.join(summary['removed_columns'])}) | "
         f"Critical Features Retained: {summary['critical_features_retained']}")
st.caption("Logged Duplicate Rows Removed:")
st.dataframe(dup_log)
st.caption("Column Relevance Audit Log:")
st.dataframe(col_log)

st.markdown('<div class="section"><h3>2. Dynamic Column Name Standardizer & Data Type Parser (Today Task 2)</h3>'
            '<p>Normalizes raw column headers into canonical snake_case and parses embedded units (yrs, mmHg, mg/dL, bpm), comma decimals, and text booleans into strict int64/float64 types with zero errors.</p></div>',
            unsafe_allow_html=True)

unformatted_df = SP.create_unformatted_sample(raw_df, n_rows=25)
parsed_df, col_map_df, dtype_log_df = SP.parse_and_cast_dtypes(unformatted_df)
st.caption("Column Name Standardization Log:")
st.dataframe(col_map_df)
st.caption("Dynamic Data Type Conversion & Repair Log:")
st.dataframe(dtype_log_df)

st.markdown('<div class="section"><h3>3. Multi-Method Outlier Detection & Case-by-Case Clinical Treatment (Today Task 3)</h3>'
            '<p>Applies IQR, Z-Score (|Z| > 3), and multivariate DBSCAN clustering across numerical columns, then executes domain-specific retain, winsorize/cap, log-transform, or anomaly-removal actions.</p></div>',
            unsafe_allow_html=True)

outlier_summary_df, _, db_total = OL.detect_outliers_all_methods(raw_df)
treated_df, decisions_df = OL.apply_domain_outlier_strategy(raw_df)
st.write(f"Multivariate DBSCAN noise/outlier patients identified across 5 continuous features: {db_total}")
st.caption("Outlier Detection Summary (IQR vs Z-Score vs DBSCAN):")
st.dataframe(outlier_summary_df)
st.caption("Case-by-Case Domain Treatment Decisions & Skewness Improvement:")
st.dataframe(decisions_df)

st.download_button("Download Cleaned & Outlier-Treated Dataset (heart_cleaned_pipeline.csv)",
                   treated_df.to_csv(index=False), "heart_cleaned_pipeline.csv", "text/csv")

cm.footer()
