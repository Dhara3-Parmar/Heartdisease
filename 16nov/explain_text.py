"""
HeartCare - template-based Natural Language Generation (NLG) for human-readable prediction justifications.

Dynamically calibrates explanation tone, length, and hedging across 4 confidence tiers (High, Moderate, Low, Toss-up),
and computes minimum-change single-feature counterfactuals.
"""
import numpy as np

from confidence import CLASS_NAMES, FEATURES, sweep, sweep_values

HUMAN = {"age": "age", "sex": "sex", "cp": "chest pain type", "trestbps": "resting blood pressure", "chol": "cholesterol",
         "fbs": "fasting blood sugar", "restecg": "resting ECG", "thalach": "maximum heart rate", "exang": "exercise-induced angina",
         "oldpeak": "ST depression (oldpeak)", "slope": "ST slope", "ca": "number of major vessels (ca)", "thal": "thal test result"}
UNIT = {"age": " yrs", "trestbps": " mmHg", "chol": " mg/dL", "thalach": " bpm"}

TIERS = {
    "high": (0.30, "&#10003;", "High confidence", "The model strongly indicates that"),
    "moderate": (0.15, "&#9679;", "Moderate confidence", "The model leans toward the conclusion that"),
    "low": (0.05, "&#9888;", "Low confidence (close call)", "The model shows a slight inclination that"),
    "toss-up": (0.0, "&#9888;", "Toss-up", "The model is nearly evenly split, with a marginal lean that"),
}
ORDER = ["high", "moderate", "low", "toss-up"]


def pct(p):
    return "<1%" if p < 0.005 else (">99%" if p > 0.995 else f"{p * 100:.0f}%")


def value_text(f, v):
    return f"{v:g}{UNIT.get(f, '')}"


def confidence_tier(prob, threshold=0.5, n_filled=0, phi=None):
    """Determine confidence tier from probability margin, imputed field count, and conflicting SHAP signals."""
    margin = abs(prob - threshold)
    base = next(t for t in ORDER if margin >= TIERS[t][0])
    reasons = []
    i = ORDER.index(base)
    if n_filled >= 2 and i < 3:
        i += 1; reasons.append(f"{n_filled} values were auto-imputed rather than measured, reducing confidence by one tier")
    if phi is not None:
        pos, neg = phi[phi > 0].sum(), -phi[phi < 0].sum()
        lead, opp = max(pos, neg), min(pos, neg)
        if lead > 0 and opp / lead >= 0.6 and i < 3:
            i += 1; reasons.append("several features pull strongly in opposing directions (mixed clinical signals)")
    return ORDER[i], reasons, base


def counterfactual(model, x, rules, threshold=0.5, features=("chol", "trestbps", "thalach", "oldpeak")):
    """Find the smallest single-feature change among continuous features that flips the classification."""
    base_p = float(sweep(model, x, features[0], [float(x[FEATURES.index(features[0])])])[0])
    want_high = base_p <= threshold
    best = None
    for f in features:
        r = rules[f]; cur = float(x[FEATURES.index(f)])
        vals = sweep_values(r, 41)
        ps = sweep(model, x, f, vals)
        for v, p in zip(vals, ps):
            crossed = (p > threshold) if want_high else (p <= threshold)
            if crossed:
                rel = abs(v - cur) / (r["max"] - r["min"])
                if best is None or rel < best["rel"]:
                    best = {"feature": f, "from": cur, "to": float(v), "p_after": float(p), "rel": rel}
    return best


def explain(info_shap, prob, threshold=0.5, filled=(), model=None, rules=None, x=None, population=None):
    """Generate structured natural-language explanation, confidence tier, verifiable facts, and counterfactual."""
    phi = info_shap["phi"]; xs = info_shap["x"]
    tier, reasons, base = confidence_tier(prob, threshold, len(filled), phi)
    decision = 1 if prob > threshold else 0
    lead_cls, other_cls = CLASS_NAMES[decision], CLASS_NAMES[1 - decision]
    sign = 1 if decision == 1 else -1
    order = np.argsort(-np.abs(phi))
    for_ = [i for i in order if np.sign(phi[i]) == sign][:2 if tier == "high" else 3]
    against = [i for i in order if np.sign(phi[i]) == -sign][:1]
    facts, sents = [], []

    sents.append(f"{TIERS[base][3]} this patient's profile is <b>{lead_cls}</b> (estimated {CLASS_NAMES[1]} probability: {pct(prob)}, "
                 f"decision threshold: {threshold * 100:.0f}%).")
    facts.append({"type": "headline", "decision": decision, "prob": prob})

    def phrase(i):
        f = FEATURES[i]; est = " (estimated value)" if f in filled else ""
        pct_str = ""
        if population is not None and f in population and f in UNIT:
            share = float((np.asarray(population[f]) < xs[i]).mean() * 100)
            pct_str = f", higher than {share:.0f}% of dataset patients" if share >= 50 else f", lower than {100 - share:.0f}% of dataset patients"
        return f"{HUMAN[f]} = {value_text(f, xs[i])}{est}{pct_str} (<b>{abs(phi[i]) * 100:.1f} pts</b>)"

    if for_:
        sents.append(f"Primary supporting factors: {'; '.join(phrase(i) for i in for_)} — pushing the prediction toward {lead_cls}.")
        facts += [{"type": "for", "feature": FEATURES[i], "pts": float(abs(phi[i]) * 100), "dir": sign} for i in for_]
    if against and tier != "high":
        i = against[0]
        sents.append(f"Counter-balancing factor: {phrase(i)} — pulling toward {other_cls}.")
        facts.append({"type": "against", "feature": FEATURES[i], "pts": float(abs(phi[i]) * 100), "dir": -sign})

    if tier == "high":
        sents.append("Multiple key features align in the same direction, making this prediction stable.")
    else:
        why = " ".join(r[0].upper() + r[1:] + "." for r in reasons)
        close = base in ("low", "toss-up")
        line = {"moderate": "The prediction leans in one direction, but with moderate uncertainty.",
                "low": "This is a close call: small changes in input values could flip the result." if close else
                       "Confidence in this prediction is reduced; interpret with caution.",
                "toss-up": "This case is near 50-50; treat the classification as inconclusive." if close else
                           "Overall confidence is low; do not rely on this single threshold decision."}[tier]
        sents.append((line + " " + why).strip())

    cf = None
    if model is not None and rules is not None and x is not None and tier != "high":
        cf = counterfactual(model, x, rules, threshold)
        if cf:
            sents.append(f"Counterfactual: according to the model, changing {HUMAN[cf['feature']]} alone from {value_text(cf['feature'], cf['from'])} to "
                         f"{value_text(cf['feature'], cf['to'])} would flip the prediction to {other_cls} ({CLASS_NAMES[1]} probability {pct(cf['p_after'])}).")
        else:
            sents.append("Changing any single continuous feature (cholesterol, blood pressure, heart rate, or oldpeak) within its normal range does not flip the decision.")
    sents.append("Note: These explanations reflect statistical patterns learned by the model, not clinical diagnosis.")
    return {"sentences": sents, "tier": tier, "reasons": reasons, "facts": facts, "counterfactual": cf}


CSS = """
<style>
.ex-card { background: var(--card, #fff); border: 1px solid var(--line, #cbd5e1); border-left-width: 6px; border-radius: 12px;
  padding: 16px 18px; margin: 12px 0; animation: ex-in .4s ease both; }
.ex-card.high { border-left-color: #1d4ed8; } .ex-card.moderate { border-left-color: #0f766e; }
.ex-card.low, .ex-card.toss-up { border-left-color: #b45309; }
.ex-badge { display: inline-block; font-weight: 700; font-size: 14px; padding: 3px 10px; border-radius: 999px; background: #e8eefb; color: #142b53; margin-bottom: 8px; }
.ex-card.low .ex-badge, .ex-card.toss-up .ex-badge { background: #fdf0d9; color: #6b3a05; }
.ex-card p { margin: 6px 0; line-height: 1.55; color: #1e293b; font-size: 16px; }
@keyframes ex-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .ex-card { animation: none; } }
</style>
"""


def render(ex):
    icon, label = TIERS[ex["tier"]][1], TIERS[ex["tier"]][2]
    body = "".join(f"<p>{s}</p>" for s in ex["sentences"])
    return f'<div class="ex-card {ex["tier"]}" role="region" aria-label="Prediction explanation"><span class="ex-badge">{icon} {label}</span>{body}</div>'
