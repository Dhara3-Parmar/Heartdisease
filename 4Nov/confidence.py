"""
HeartCare - prediction confidence ko bar charts me dikhana.

1. Probability distribution HAATH se: Random Forest ki har tree ka probability nikal ke khud average karte hain
   (model.predict_proba ka use sirf check karne ke liye).
2. Charts HTML + CSS se bante hain. Har bar purani value se nayi value tak animate hota hai (CSS keyframes).
3. Interactive: threshold slider, what-if (ek feature badalke confidence kaise badalti hai).
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
CLASS_NAMES = {0: "Lower Risk", 1: "Higher Risk"}       
COLORS = {0: "#177245", 1: "#b42323"}          
N_BINS = 10


def analyse(model, x):
    """Ek patient ke liye poori confidence picture. Cache me yahi dict jati hai."""
    x32 = np.ascontiguousarray(x, dtype=np.float32).reshape(1, -1)             
    per_tree = np.array([t.predict_proba(x32, check_input=False)[0] for t in model.estimators_])
    probs = per_tree.mean(axis=0)                                                  
    votes = np.bincount(per_tree.argmax(axis=1), minlength=len(probs))           
    hist = np.histogram(per_tree[:, 1], bins=N_BINS, range=(0, 1))[0]              
    return {"probs": probs, "votes": votes, "hist": hist, "n_trees": len(per_tree),
            "std": float(per_tree[:, 1].std()), "x": np.asarray(x, dtype=float)}


def decide(probs, threshold=0.5):
    """Threshold ke hisaab se class (default 0.5 = model.predict jaisa)."""
    return int(probs[1] > threshold)


def sweep(model, x, feature, values):
    """What-if: ek feature ki value badalke P(class 1) (ek hi predict_proba call me)."""
    rows = np.tile(np.asarray(x, dtype=float), (len(values), 1))
    rows[:, FEATURES.index(feature)] = values
    return model.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]


def sweep_values(rule, n=10):
    """Rules (input.py) se feature ki values: category = allowed, number = n barabar points."""
    if rule["kind"] == "category":
        return [float(v) for v in rule["allowed"]]
    return [float(v) for v in np.round(np.linspace(rule["min"], rule["max"], n), 1)]


CSS = """
<style>
.hc-chart { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 14px; }
.hc-chart h4 { margin: 0 0 12px 0; color: #142b53; font-size: 16px; }
.hc-row { display: flex; align-items: center; margin: 8px 0; }
.hc-label { width: 120px; font-size: 14px; color: #334155; }
.hc-track { position: relative; flex: 1; height: 22px; background: #eef2f7; border-radius: 6px; overflow: hidden; }
.hc-fill { height: 100%; width: var(--to); animation: hc-grow 0.9s ease-out; }
.hc-value { width: 64px; text-align: right; font-size: 14px; font-weight: 600; color: #142b53; }
.hc-thr { position: absolute; top: 0; bottom: 0; width: 2px; background: #142b53; }
.hc-cols { display: flex; align-items: flex-end; height: 130px; gap: 6px; border-bottom: 1px solid #cbd5e1; }
.hc-colwrap { flex: 1; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; }
.hc-col { width: 100%; height: var(--to); border-radius: 4px 4px 0 0; animation: hc-rise 0.9s ease-out; }
.hc-colnum { font-size: 11px; color: #475569; height: 14px; }
.hc-xlabels { display: flex; gap: 6px; margin-top: 4px; }
.hc-xlabels span { flex: 1; text-align: center; font-size: 11px; color: #475569; }
.hc-note { font-size: 13px; color: #64748b; margin-top: 8px; }
@keyframes hc-grow { from { width: var(--from); } to { width: var(--to); } }
@keyframes hc-rise { from { height: var(--from); } to { height: var(--to); } }
@media (max-width: 600px) { .hc-label { width: 84px; font-size: 12px; } .hc-value { width: 52px; } }
@media (prefers-reduced-motion: reduce) { .hc-fill, .hc-col { animation: none; } }
</style>
"""


def _prev(prev, i, scale):
    """Purani value (animation ka start). Pehli baar 0 se grow hota hai."""
    return 0.0 if prev is None or i >= len(prev) else float(prev[i]) * scale


def bar_chart(title, labels, values, prev=None, colors=None, threshold=None, note=""):
    """Horizontal bars. values 0-1 (probability). threshold = vertical line."""
    rows = ""
    for i, (lab, v) in enumerate(zip(labels, values)):
        color = colors[i] if colors else "#244577"
        thr = f'<div class="hc-thr" style="left:{threshold * 100:.1f}%"></div>' if threshold is not None else ""
        rows += (f'<div class="hc-row"><div class="hc-label">{lab}</div>'
                 f'<div class="hc-track"><div class="hc-fill" style="--from:{_prev(prev, i, 100):.1f}%;--to:{v * 100:.1f}%;background:{color}"></div>{thr}</div>'
                 f'<div class="hc-value">{v * 100:.1f}%</div></div>')
    note_html = f'<div class="hc-note">{note}</div>' if note else ""
    return f'<div class="hc-chart"><h4>{title}</h4>{rows}{note_html}</div>'


def column_chart(title, x_labels, values, prev=None, color="#244577", counts=None, note=""):
    """Vertical bars. values 0-1 (height ka fraction); counts = bar ke upar likha number."""
    cols, labs = "", ""
    for i, (lab, v) in enumerate(zip(x_labels, values)):
        shown = counts[i] if counts is not None else f"{v * 100:.0f}%"
        cols += (f'<div class="hc-colwrap"><div class="hc-colnum">{shown}</div>'
                 f'<div class="hc-col" style="--from:{_prev(prev, i, 100):.1f}%;--to:{v * 100:.1f}%;background:{color}"></div></div>')
        labs += f"<span>{lab}</span>"
    note_html = f'<div class="hc-note">{note}</div>' if note else ""
    return f'<div class="hc-chart"><h4>{title}</h4><div class="hc-cols">{cols}</div><div class="hc-xlabels">{labs}</div>{note_html}</div>'
