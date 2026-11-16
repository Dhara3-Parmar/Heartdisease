"""Page 2: Explain - natural language explanation, exact SHAP values, and interactive risk curve."""
import streamlit as st

import common as cm
import components as ui
import explain_text as ex
import shap_values as sv
import theme
from confidence import CLASS_NAMES, FEATURES, analyse, sweep, sweep_values

cm.page_setup("Explain", "Why did the model make this prediction?")
model, rules, cache, model_version, data_df, background = cm.load()
check = cm.need_check()

x = check["x"]; filled = list(check["filled"])
threshold = st.session_state.get("thr", 0.5)
xkey = tuple(float(v) for v in x)

info = cache.get_or_compute((model_version, xkey), lambda: analyse(model, x))
p1 = float(info["probs"][1])
sinfo = cache.get_or_compute((model_version, "shap", xkey, len(background)), lambda: sv.explain(model, x, background))
population = {f: data_df[f].values for f in ("age", "trestbps", "chol", "thalach")}
nlg = cache.get_or_compute((model_version, "nlg", xkey, threshold, tuple(filled)),
                           lambda: ex.explain(sinfo, p1, threshold, filled, model, rules, x, population))

st.markdown('<div class="section"><h3>Human-Readable Explanation (NLG)</h3><p>Generated directly from exact Shapley attributions and calibrated to model confidence.</p></div>', unsafe_allow_html=True)
st.markdown(theme.risk_badge(1 if p1 > threshold else 0, (CLASS_NAMES[0], CLASS_NAMES[1])), unsafe_allow_html=True)
st.markdown(ex.render(nlg), unsafe_allow_html=True)

st.markdown('<div class="section"><h3>Exact SHAP Attributions (Percentage Points)</h3><p>Shows how much each feature shifted the risk from the baseline patient average across all 2^13 coalitions.</p></div>', unsafe_allow_html=True)
st.session_state.setdefault("top_n", 8)
st.slider("SHAP: number of top features to display", 3, 13, key="top_n")
inspect = st.selectbox("Inspect a single feature in detail", ["(none)"] + FEATURES, key="inspect")
sel = None if inspect == "(none)" else inspect
old_phi = cm.previous("shap", sinfo["phi"])
shap_prev = dict(zip(FEATURES, old_phi)) if old_phi else None
st.markdown(sv.shap_chart(sinfo, st.session_state["top_n"], sel, prev=shap_prev, filled=filled), unsafe_allow_html=True)
if sel:
    st.markdown(sv.feature_detail(sinfo, sel, background.mean(axis=0), filled), unsafe_allow_html=True)
st.caption("Transparency note: these attributions reflect model patterns, not clinical causation. Baseline = average of 20 reference patients.")

st.markdown('<div class="section"><h3>Risk Sensitivity Curve (What-If)</h3><p>Vary one feature while keeping all others fixed.</p></div>', unsafe_allow_html=True)
explore = st.selectbox("Select feature for risk curve", FEATURES, key="explore")
r = rules[explore]; fi = FEATURES.index(explore)
xs = sweep_values(r, 25) if r["kind"] == "number" else sweep_values(r)
ps = sweep(model, x, explore, xs)
st.markdown(ui.curve_chart(f"Risk curve: effect of varying {explore}", explore, xs, ps, float(x[fi]), p1,
                           data_df[explore].values, r["kind"] == "category", threshold), unsafe_allow_html=True)
cm.nav_link("pages/1_Predict.py", "Return to Predict Page to modify inputs")
cm.footer()
