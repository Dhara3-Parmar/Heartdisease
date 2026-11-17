"""Page 4: Feedback - usability feedback collection form and summary dashboard."""
import streamlit as st

import common as cm
from feedback import add_feedback, load_feedback, summarize

cm.page_setup("Feedback", "Share your usability feedback to help improve HeartCare")
cm.load()

st.markdown('<div class="section"><h3>Submit Feedback</h3></div>', unsafe_allow_html=True)
device = st.radio("Which device are you using?", ["Phone", "Tablet", "Laptop"], horizontal=True, key="fb_device")
task = st.radio("Were you able to complete your task without assistance?", ["Yes", "Partly", "No"], horizontal=True, key="fb_task")
ease = st.slider("How easy was the app to use? (1 = difficult, 5 = very easy)", 1, 5, 4, key="fb_ease")
text = st.text_area("Any issues or suggestions? (optional)", key="fb_text", max_chars=500)
if st.button("Send feedback"):
    try:
        add_feedback(cm.FEEDBACK_FILE, device, task, ease, text)
        st.success("Thank you! Your feedback has been saved.")
    except Exception as e:
        st.error(f"Could not save feedback: {e}")

st.markdown('<div class="section"><h3>Feedback Summary</h3></div>', unsafe_allow_html=True)
s = summarize(load_feedback(cm.FEEDBACK_FILE))
if s["n"] == 0:
    st.caption("No feedback entries recorded yet.")
else:
    st.write(f"Responses: {s['n']} | Task success: {s['success_rate_%']}% | Average ease: {s['mean_ease']}/5")
    if s["n"] < 5:
        st.caption("Fewer than 5 responses collected so far; treat summary statistics as preliminary.")
    if s["themes"]:
        st.caption("Comment themes: " + ", ".join(f"{k} ({v})" for k, v in s["themes"].items()))
cm.footer()
