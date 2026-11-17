"""Page 1: Predict - interactive sliders, quick text entry, validation framework, live risk card, and confidence charts."""
import time

import streamlit as st

import common as cm
import components as ui
import theme
import validation as V
from confidence import CLASS_NAMES, COLORS, FEATURES, analyse, bar_chart, column_chart, decide

cm.page_setup("Predict", "Enter patient clinical details for real-time risk assessment")
model, rules, cache, model_version, data_df, background = cm.load()


def apply_values(clean):
    notes = []
    for f, v in clean.items():
        if v is None:
            continue
        r = rules[f]
        if r["kind"] == "number" and not (r["min"] <= v <= r["max"]):
            near = min(max(v, r["min"]), r["max"])
            notes.append(V.issue(f, "warning", f"{f}: {v:g} is outside the slider range ({r['min']:g} to {r['max']:g}); clamped slider to {near:g}. "
                                 f"Use the Batch (CSV) page for extreme out-of-range values."))
        cm.set_input(rules, f, v)
    return notes


def apply_quick():
    record, issues = V.parse_quick(st.session_state.get("quick", ""))
    d = V.diagnose(record, rules, model, partial=True)
    st.session_state["qissues"] = issues + d["issues"] + apply_values(d["clean"])


def take_fix(idx):
    issues = st.session_state.get("qissues", [])
    if idx >= len(issues):
        return
    chosen = issues[idx]
    record = V.apply_fix({}, chosen)
    d = V.diagnose(record, rules, model, partial=True)
    rest = [i for k, i in enumerate(issues) if k != idx]
    st.session_state["qissues"] = rest + d["issues"] + apply_values(d["clean"])


def show_issues(issues, prefix):
    for k, i in enumerate(issues):
        if i["level"] == "error":
            st.error(i["message"])
            if i["fix"] is not None and prefix == "q":
                st.button(i["fix_label"] or "Apply suggestion", key=f"fix_{k}", on_click=take_fix, args=(k,))
        elif i["level"] == "warning":
            st.warning(i["message"])
        else:
            st.info("Auto-corrected: " + i["message"])


top = st.empty()

st.markdown('<div class="section"><h3>Quick Entry (Optional)</h3><p>Type values on a single line, e.g. <b>age 63, chol 5.2 mmol/L, bp 130, oldpeak 1,5</b>. '
            'Field typos, comma decimals, and units are handled automatically.</p></div>', unsafe_allow_html=True)
st.text_input("Quick entry", key="quick", placeholder="age 63, chol 240, ca 2", label_visibility="collapsed")
st.button("Apply quick entry", on_click=apply_quick)
show_issues(st.session_state.get("qissues", []), "q")

values = {}
for title, feats in cm.SECTIONS:
    st.markdown(f'<div class="section"><h3>{title}</h3></div>', unsafe_allow_html=True)
    for f in feats:
        values[f] = cm.widget(rules, data_df, f)

unknown = st.multiselect("Values I don't know (max 3)", [cm.LABELS[f] for f in FEATURES], key="unknown")
too_many = len(unknown) > cm.MAX_UNKNOWN
if too_many:
    st.error(f"Maximum {cm.MAX_UNKNOWN} values can be marked as unknown. Please remove one or two.")

live = st.checkbox("Live update (update prediction automatically as sliders move)", value=True)
clicked = st.button("Predict Heart Disease Risk")
st.button("Reset to typical values", on_click=cm.reset_inputs, args=(rules,))

if (clicked or live) and not too_many:
    record = {f: (None if cm.LABELS[f] in unknown else values[f]) for f in FEATURES}
    check = V.check_record(record, rules, model)
    show_issues([i for i in check["issues"] if i["level"] == "error"], "form")
    if check["ok"]:
        st.session_state["check"] = check
    else:
        st.session_state.pop("check", None)

if too_many:
    st.session_state.pop("check", None)

check = st.session_state.get("check")

if check:
    st.markdown('<div class="section"><h3>Prediction & Confidence</h3></div>', unsafe_allow_html=True)
    st.session_state.setdefault("threshold", 50)
    st.slider("Decision threshold (%): classify as Higher Risk when probability exceeds this value", 1, 99, key="threshold")
    threshold = st.session_state["threshold"] / 100
    st.session_state["thr"] = threshold

    start = time.perf_counter()
    key = (model_version, tuple(float(v) for v in check["x"]))
    info = cache.get_or_compute(key, lambda: analyse(model, check["x"]))
    elapsed = (time.perf_counter() - start) * 1000

    for message in check["warnings"]:
        st.warning(message)

    filled = list(check["filled"])
    probs = info["probs"]
    p1 = float(probs[1])
    prev_card = st.session_state.get("last_p1")
    st.session_state["last_p1"] = p1
    top.markdown(ui.risk_card(p1, threshold, prev_card, filled), unsafe_allow_html=True)

    prediction = decide(probs, threshold)
    st.markdown(theme.risk_badge(prediction, (CLASS_NAMES[0], CLASS_NAMES[1])), unsafe_allow_html=True)
    st.write("Model Confidence:", round(max(probs) * 100, 2), "%")
    if threshold != 0.5:
        st.caption(f"Custom threshold {threshold * 100:.0f}%: {CLASS_NAMES[1]} probability is {probs[1] * 100:.1f}%, "
                   f"so prediction is {CLASS_NAMES[prediction]} (default threshold is 50%).")
    st.caption(f"{info['votes'][1]} of {info['n_trees']} trees voted for {CLASS_NAMES[1]} | inter-tree spread (std): {info['std']:.2f}")

    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    st.markdown(bar_chart("Class probabilities (ensemble average)", labels, probs,
                          prev=cm.previous("probs", probs), colors=[COLORS[0], COLORS[1]], threshold=threshold,
                          note=f"Vertical line = threshold ({threshold * 100:.0f}%)"), unsafe_allow_html=True)
    votes = info["votes"] / info["n_trees"]
    st.markdown(bar_chart("Tree vote proportions", labels, votes,
                          prev=cm.previous("votes", votes), colors=[COLORS[0], COLORS[1]]), unsafe_allow_html=True)
    hist = info["hist"]
    bins = [f"{i * 10}-{i * 10 + 10}" for i in range(len(hist))]
    st.markdown(column_chart(f"Per-tree {CLASS_NAMES[1]} probability distribution (%)", bins, hist / hist.max(),
                             prev=cm.previous("hist", hist / hist.max()), color="#244577", counts=[int(h) for h in hist],
                             note="Number above each bar shows tree count."), unsafe_allow_html=True)

    if not live:
        st.caption("Showing result from previous Predict click. Click Predict again or enable Live update.")
    cm.nav_link("pages/2_Explain.py", "Why this prediction? Open Explain Page")
    st.caption(f"Request latency: {elapsed:.2f} ms")

cm.footer()
