"""Page 8: Final Project Documentation - Methodologies, Results, Challenges & Recommendations (Today Task 2)."""
import streamlit as st

import common as cm
import documentation as DOC

cm.page_setup("Project Documentation", "Methodologies, experimental results, deployment challenges, and future recommendations")
cm.load()

st.markdown('<div class="section"><h3>1. Summary of Methodologies</h3></div>', unsafe_allow_html=True)
st.dataframe(DOC.methodology_table())

st.markdown('<div class="section"><h3>2. Experimental Results</h3></div>', unsafe_allow_html=True)
st.dataframe(DOC.experimental_results_table())

st.markdown('<div class="section"><h3>3. Deployment Challenges & Engineering Solutions</h3></div>', unsafe_allow_html=True)
st.dataframe(DOC.challenges_table())

st.markdown('<div class="section"><h3>4. Actionable Recommendations for Future Improvements</h3></div>', unsafe_allow_html=True)
st.dataframe(DOC.recommendations_table())

st.download_button("Download Full Documentation (Markdown)", DOC.generate_markdown_report(), "PROJECT_DOCUMENTATION.md", "text/markdown")
cm.footer()
