"""
HeartCare - custom visualization components (HTML + inline SVG + CSS, koi chart library nahi).

- risk_card   : live result card (sliders hilte hi update; purani value se nayi tak animate)
- curve_chart : interactive risk curve (SVG): population histogram + model ka risk curve + "aap yahan hain" marker
- percentile_text : slider ke neeche "aap dataset me kahan hain"

Multiple devices: SVG viewBox se chart apne aap scale hota hai, cards flex-wrap karte hain,
fonts mobile par chhote nahi hote, animations 'reduced motion' setting ko maante hain.
"""
import numpy as np

from confidence import CLASS_NAMES, COLORS

CSS = """
<style>
.ui-card { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 18px; margin-bottom: 14px; }
.ui-top { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px; }
.ui-big { font-size: 40px; font-weight: 700; line-height: 1.1; color: #142b53; }
.ui-label { font-size: 16px; font-weight: 600; }
.ui-sub { font-size: 13px; color: #64748b; }
.ui-meter { position: relative; height: 14px; border-radius: 7px; margin-top: 14px; background: linear-gradient(90deg, #2e9e6b, #f2c94c, #d64545); }
.ui-pin { position: absolute; top: -5px; width: 6px; height: 24px; border-radius: 3px; background: #142b53; left: var(--to); margin-left: -3px; animation: ui-move 0.8s ease-out; }
.ui-thr { position: absolute; top: -2px; bottom: -2px; width: 2px; background: #475569; opacity: .7; }
.ui-scale { display: flex; justify-content: space-between; font-size: 11px; color: #64748b; margin-top: 4px; }
.ui-svg { width: 100%; height: auto; max-width: 640px; display: block; margin: 0 auto; }
.ui-svg text { font-family: inherit; fill: #475569; }
.ui-line { fill: none; stroke: #d64545; stroke-width: 2.5; stroke-linejoin: round; stroke-dasharray: 1; stroke-dashoffset: 0; animation: ui-draw 1s ease-out; }
.ui-pt:hover { r: 6; }
@keyframes ui-move { from { left: var(--from); } to { left: var(--to); } }
@keyframes ui-draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }
@media (max-width: 600px) { .ui-big { font-size: 32px; } .ui-card { padding: 12px; } }
@media (prefers-reduced-motion: reduce) { .ui-pin, .ui-line { animation: none; } }
</style>
"""


def risk_card(prob, threshold=0.5, prev_prob=None, estimated=()):
    """Top par live card: P(Higher Risk), decision, pichle update se farak, meter."""
    decision = 1 if prob > threshold else 0
    start = prob if prev_prob is None else prev_prob
    delta = "" if prev_prob is None or abs(prob - prev_prob) < 1e-9 else \
        f'<span class="ui-sub">{"&#9650;" if prob > prev_prob else "&#9660;"} {abs(prob - prev_prob) * 100:.1f} pts pichhle update se</span>'
    est = f'<div class="ui-sub">Estimated (aapne nahi diya): {", ".join(estimated)}</div>' if estimated else ""
    return (f'<div class="ui-card"><div class="ui-top"><div class="ui-big">{prob * 100:.1f}%</div>'
            f'<div><div class="ui-label" style="color:{COLORS[decision]}">{CLASS_NAMES[decision]}</div>'
            f'<div class="ui-sub">{CLASS_NAMES[1]} probability (threshold {threshold * 100:.0f}%)</div>{delta}</div></div>'
            f'<div class="ui-meter"><div class="ui-thr" style="left:{threshold * 100:.1f}%"></div>'
            f'<div class="ui-pin" style="--from:{start * 100:.1f}%;--to:{prob * 100:.1f}%"></div></div>'
            f'<div class="ui-scale"><span>0%</span><span>50%</span><span>100%</span></div>{est}</div>')


def curve_chart(title, feature, xs, ps, current_x, current_p, population, is_category, threshold=0.5):
    """SVG risk curve. xs/ps = feature ki values aur model ka P(Higher Risk); population = dataset ki us feature ki values."""
    W, H, L, R, T, B = 380, 240, 44, 10, 14, 46
    pw, ph = W - L - R, H - T - B
    xs = np.asarray(xs, float); ps = np.asarray(ps, float)
    lo, hi = xs.min(), xs.max()
    sx = (lambda v: L + (np.where(np.asarray(xs) == v)[0][0] + 0.5) / len(xs) * pw) if is_category else \
         (lambda v: L + (float(v) - lo) / (hi - lo + 1e-9) * pw)
    sy = lambda p: T + (1 - float(p)) * ph
    out = [f'<svg class="ui-svg" viewBox="0 0 {W} {H}" role="img" aria-label="{title}">']
    for g in (0, .25, .5, .75, 1):                                                  
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(g):.1f}" y2="{sy(g):.1f}" stroke="#e2e8f0"/><text x="{L - 6}" y="{sy(g) + 4:.1f}" font-size="13" text-anchor="end">{g * 100:.0f}%</text>')
    pop = np.asarray(population, float)                                           
    if is_category:
        counts = np.array([(pop == v).sum() for v in xs]); edges = None
    else:
        counts, edges = np.histogram(pop, bins=12, range=(lo, hi))
    top = counts.max() or 1
    for i, c in enumerate(counts):
        if is_category:
            x0, bw = L + i / len(xs) * pw + 4, pw / len(xs) - 8
        else:
            x0, bw = L + (edges[i] - lo) / (hi - lo + 1e-9) * pw + 1, (edges[i + 1] - edges[i]) / (hi - lo + 1e-9) * pw - 2
        bh = c / top * ph * 0.35
        out.append(f'<rect x="{x0:.1f}" y="{T + ph - bh:.1f}" width="{max(bw, 1):.1f}" height="{bh:.1f}" fill="#cbd5e1" opacity=".6"><title>{int(c)} patients</title></rect>')
    ty = sy(threshold)                                                             
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="#475569" stroke-dasharray="4 4"/><text x="{W - R}" y="{ty - 4:.1f}" font-size="12" text-anchor="end">threshold</text>')
    pts = " ".join(f"{sx(x):.1f},{sy(p):.1f}" for x, p in zip(xs, ps))
    out.append(f'<polyline class="ui-line" pathLength="1" points="{pts}"/>')
    for x, p in zip(xs, ps):                                                        
        out.append(f'<circle class="ui-pt" cx="{sx(x):.1f}" cy="{sy(p):.1f}" r="4" fill="#d64545"><title>{feature} = {x:g}: {CLASS_NAMES[1]} {p * 100:.1f}%</title></circle>')
    cx = sx(xs[np.abs(xs - current_x).argmin()]) if is_category else sx(min(max(current_x, lo), hi))  
    out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{T}" y2="{T + ph}" stroke="#142b53" stroke-width="2"/>'
               f'<circle cx="{cx:.1f}" cy="{sy(current_p):.1f}" r="7" fill="#142b53" stroke="white" stroke-width="2"><title>Current: {feature} = {current_x:g}, {current_p * 100:.1f}%</title></circle>')
    step = max(1, len(xs) // 6)                                                    
    for i in range(0, len(xs), step):
        out.append(f'<text x="{sx(xs[i]):.1f}" y="{H - 26}" font-size="13" text-anchor="middle">{xs[i]:g}</text>')
    out.append(f'<text x="{L + pw / 2}" y="{H - 6}" font-size="13" text-anchor="middle">{feature} (grey = dataset ke patients)</text></svg>')
    return f'<div class="hc-chart"><h4>{title}</h4>{"".join(out)}</div>'


def percentile_text(value, population):
    """Slider value ka dataset me position."""
    pop = np.asarray(population, float)
    pct = (pop < value).mean() * 100
    return f"Dataset me {pct:.0f}% patients se zyada (median {np.median(pop):g})"
