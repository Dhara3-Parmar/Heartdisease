"""
HeartCare - SHAP values (haath se) aur unka interactive display.

Shapley value: ek feature ka "fair share" jo prediction ko average patient se door le jata hai.
  phi_i = sum over coalitions S (i ke bina)  |S|! (M-|S|-1)! / M!  *  [ v(S + i) - v(S) ]
  v(S)  = S ke features patient se, baaki features background (reference) patients se; model ka average output.
Yahan koi `shap` library nahi. 13 features = 2^13 = 8192 coalitions, saare exactly evaluate hote hain.
Contributions "probability points" (percentage points) me hain: P(class 1) = P(Higher Risk).
"""
from math import factorial
import numpy as np
import pandas as pd

from confidence import FEATURES, CLASS_NAMES, COLORS

M = len(FEATURES)
MASKS = ((np.arange(2 ** M)[:, None] >> np.arange(M)) & 1).astype(bool)   
SIZES = MASKS.sum(axis=1)
WEIGHTS = np.array([factorial(s) * factorial(M - s - 1) / factorial(M) for s in range(M)])


def build_background(df, k=20, seed=0):
    """Reference patients: training data se k random rows."""
    return df[FEATURES].sample(k, random_state=seed).to_numpy(dtype=float)


def coalition_values(model, x, background):
    """v(S) har coalition ke liye (8192 values): model ka average P(class 1)."""
    k = len(background)
    X = np.where(MASKS[None, :, :], np.asarray(x, float)[None, None, :], background[:, None, :])   
    p = model.predict_proba(pd.DataFrame(X.reshape(-1, M), columns=FEATURES))[:, 1]
    return p.reshape(k, -1).mean(axis=0)


def shapley_exact(model, x, background):
    """Exact Shapley values. Return: phi (13,), base value, prediction f(x)."""
    v = coalition_values(model, x, background)
    phi = np.zeros(M)
    for i in range(M):
        without = np.where(~MASKS[:, i])[0]                    
        phi[i] = (WEIGHTS[SIZES[without]] * (v[without | (1 << i)] - v[without])).sum()
    return phi, float(v[0]), float(v[-1])                      


def shapley_sampling(model, x, background, n_perm=200, seed=0):
    """Doosra independent tareeka (verification ke liye): random orderings me feature ek-ek karke patient wali value par laate hain."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    phi = np.zeros(M)
    for _ in range(n_perm):
        order = rng.permutation(M)
        b = background[rng.integers(len(background))].copy()
        rows = [b.copy()]
        for j in order:
            b[j] = x[j]; rows.append(b.copy())
        p = model.predict_proba(pd.DataFrame(np.array(rows), columns=FEATURES))[:, 1]
        phi[order] += np.diff(p)
    return phi / n_perm


def explain(model, x, background):
    phi, base, fx = shapley_exact(model, x, background)
    return {"phi": phi, "base": base, "pred": fx, "x": np.asarray(x, float), "k": len(background)}


def fmt(v):
    return f"{v:g}"


def plain_summary(info, filled=()):
    """Top drivers ek line me, aam bhasha me."""
    order = np.argsort(-np.abs(info["phi"]))[:3]
    parts = []
    for i in order:
        f = FEATURES[i]; pts = info["phi"][i] * 100
        direction = CLASS_NAMES[1] if pts > 0 else CLASS_NAMES[0]
        est = " (estimated)" if f in filled else ""
        parts.append(f"{f} = {fmt(info['x'][i])}{est} {direction} ki taraf {abs(pts):.1f} pts")
    return "Sabse bade kaaran: " + "; ".join(parts) + "."


def feature_detail(info, feature, bg_mean, filled=()):
    """Ek feature ka detail: value, average patient se compare, contribution."""
    i = FEATURES.index(feature); pts = info["phi"][i] * 100
    side = f"{CLASS_NAMES[1]} ki taraf" if pts > 0 else f"{CLASS_NAMES[0]} ki taraf"
    est = " Ye value aapne nahi di, isliye typical value se bhari gayi hai." if feature in filled else ""
    return (f"<b>{feature} = {fmt(info['x'][i])}</b> (reference patients ka average {bg_mean[i]:.1f}). "
            f"Is feature ne prediction ko <b>{abs(pts):.1f} percentage points {side}</b> dhakela.{est}")


SHAP_CSS = """
<style>
.sv-row { display: flex; align-items: center; margin: 6px 0; padding: 2px 4px; border-radius: 6px; }
.sv-row.sel { background: #eef4ff; outline: 2px solid #244577; }
.sv-label { width: 128px; font-size: 14px; color: #334155; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sv-track { position: relative; flex: 1; height: 22px; background: #f1f5f9; border-radius: 6px; }
.sv-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: #94a3b8; }
.sv-bar { position: absolute; top: 3px; bottom: 3px; width: var(--to); animation: sv-grow 0.9s ease-out; }
.sv-bar.pos { left: 50%; border-radius: 0 5px 5px 0; }
.sv-bar.neg { right: 50%; border-radius: 5px 0 0 5px; }
.sv-val { width: 78px; text-align: right; font-size: 13px; font-weight: 600; color: #142b53; }
.sv-total { display: flex; justify-content: space-between; font-size: 14px; padding: 8px 4px 0 4px; color: #142b53; border-top: 1px solid #e2e8f0; margin-top: 8px; flex-wrap: wrap; gap: 6px; }
.sv-legend { font-size: 12px; color: #475569; margin-top: 6px; }
@keyframes sv-grow { from { width: var(--from); } to { width: var(--to); } }
@media (max-width: 600px) { .sv-label { width: 92px; font-size: 12px; } .sv-val { width: 62px; font-size: 12px; } }
@media (prefers-reduced-motion: reduce) { .sv-bar { animation: none; } }
</style>
"""


def shap_chart(info, top_n=8, selected=None, prev=None, filled=()):
    """Diverging bars: kaun sa feature prediction ko kis taraf aur kitna dhakel raha hai.
    prev = {feature: pichli phi} (animation ka start)."""
    phi = info["phi"]; order = np.argsort(-np.abs(phi)); top = order[:top_n]; rest = order[top_n:]
    scale = max(np.abs(phi[order[0]]), 1e-9)                                   
    rows = ""
    for i in top:
        f = FEATURES[i]; v = phi[i]; cls = "pos" if v > 0 else "neg"
        old = (prev or {}).get(f)
        start = abs(old) / scale * 50 if old is not None and np.sign(old) == np.sign(v) else 0.0
        color = COLORS[1] if v > 0 else COLORS[0]
        est = "*" if f in filled else ""
        sel = " sel" if f == selected else ""
        rows += (f'<div class="sv-row{sel}" title="{f}: {v * 100:+.2f} percentage points"><div class="sv-label">{f} = {fmt(info["x"][i])}{est}</div>'
                 f'<div class="sv-track"><div class="sv-mid"></div><div class="sv-bar {cls}" style="--from:{start:.1f}%;--to:{abs(v) / scale * 50:.1f}%;background:{color}"></div></div>'
                 f'<div class="sv-val">{v * 100:+.1f} pts</div></div>')
    if len(rest):
        other = phi[rest].sum() * 100
        rows += (f'<div class="sv-row"><div class="sv-label">baaki {len(rest)} features</div><div class="sv-track"><div class="sv-mid"></div></div>'
                 f'<div class="sv-val">{other:+.1f} pts</div></div>')
    total = (f'<div class="sv-total"><span>Average patient: <b>{info["base"] * 100:.1f}%</b></span>'
             f'<span>+ contributions: <b>{phi.sum() * 100:+.1f} pts</b></span><span>= Is patient: <b>{info["pred"] * 100:.1f}%</b></span></div>')
    legend = (f'<div class="sv-legend"><span style="color:{COLORS[1]}">&#9632;</span> {CLASS_NAMES[1]} ki taraf &nbsp; '
              f'<span style="color:{COLORS[0]}">&#9632;</span> {CLASS_NAMES[0]} ki taraf &nbsp; (* = estimated value)</div>')
    return f'<div class="hc-chart"><h4>Kis feature ne prediction ko kitna dhakela (SHAP values)</h4>{rows}{total}{legend}</div>'
