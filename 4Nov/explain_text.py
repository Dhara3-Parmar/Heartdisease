"""
HeartCare - prediction ko aam bhasha me samjhana (natural language generation).

Method: template-based NLG (jaan-boojhkar, taaki har shabd data se verify ho sake, koi "banawati" claim na ho).
  1. Content selection : SHAP se top drivers (prediction ki taraf + khilaf), user ne jo values nahi di unka zikr
  2. Confidence tier   : margin (probability threshold se kitni door), estimated values, mixed signals -> 4 tier
  3. Sentence planning : tier ke hisaab se lambai aur hedging badalti hai (high = chhota aur seedha; borderline = zyada savdhaan)
  4. Counterfactual    : ek value badalne se prediction palatti hai ya nahi (model ke hisaab se)
Har sentence ke saath 'facts' bhi lautta hai, taaki test kar sakein ki likha hua sach hai.
"""
import numpy as np

from confidence import CLASS_NAMES, FEATURES, sweep, sweep_values

HUMAN = {"age": "age", "sex": "sex", "cp": "chest pain type", "trestbps": "resting blood pressure", "chol": "cholesterol",
         "fbs": "fasting blood sugar", "restecg": "resting ECG", "thalach": "maximum heart rate", "exang": "exercise-induced angina",
         "oldpeak": "ST depression (oldpeak)", "slope": "ST slope", "ca": "number of major vessels (ca)", "thal": "thal test result"}
UNIT = {"age": " saal", "trestbps": " mmHg", "chol": " mg/dl", "thalach": " bpm"}

TIERS = {  # name: (min margin, icon, label, headline verb)
    "high": (0.30, "&#10003;", "High confidence", "model ko kaafi bharosa hai ki"),
    "moderate": (0.15, "&#9679;", "Moderate confidence", "model ka jhukaav is taraf hai ki"),
    "low": (0.05, "&#9888;", "Low confidence (close call)", "model ko thoda hi jhukaav hai ki"),
    "toss-up": (0.0, "&#9888;", "Toss-up", "model lagbhag barabar soch raha hai, halka sa jhukaav ye hai ki"),
}
ORDER = ["high", "moderate", "low", "toss-up"]


def pct(p):
    return "<1%" if p < 0.005 else (">99%" if p > 0.995 else f"{p * 100:.0f}%")


def value_text(f, v):
    return f"{v:g}{UNIT.get(f, '')}"


def confidence_tier(prob, threshold=0.5, n_filled=0, phi=None):
    """Return (tier, reasons, base). base = sirf margin se tier; tier = estimated values / mixed signals se ek level neeche."""
    margin = abs(prob - threshold)
    base = next(t for t in ORDER if margin >= TIERS[t][0])
    reasons = []
    i = ORDER.index(base)
    if n_filled >= 2 and i < 3:
        i += 1; reasons.append(f"{n_filled} values aapne nahi di (estimated), isliye confidence ek level kam maani gayi")
    if phi is not None:
        pos, neg = phi[phi > 0].sum(), -phi[phi < 0].sum()
        lead, opp = max(pos, neg), min(pos, neg)
        if lead > 0 and opp / lead >= 0.6 and i < 3:
            i += 1; reasons.append("kuch features ek taraf aur kuch doosri taraf khinch rahe hain (mixed signals)")
    return ORDER[i], reasons, base


def counterfactual(model, x, rules, threshold=0.5, features=("chol", "trestbps", "thalach", "oldpeak")):
    """Kisi ek number feature ko sabse kam badalkar decision palatti hai? Return dict ya None. (Model ka pattern, medical salah nahi.)"""
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
    """Return {'sentences': [...], 'tier', 'reasons', 'facts': [...], 'counterfactual'}."""
    phi = info_shap["phi"]; xs = info_shap["x"]
    tier, reasons, base = confidence_tier(prob, threshold, len(filled), phi)
    decision = 1 if prob > threshold else 0
    lead_cls, other_cls = CLASS_NAMES[decision], CLASS_NAMES[1 - decision]
    sign = 1 if decision == 1 else -1                    # decision ki taraf wale drivers ka SHAP sign
    order = np.argsort(-np.abs(phi))
    for_ = [i for i in order if np.sign(phi[i]) == sign][:2 if tier == "high" else 3]
    against = [i for i in order if np.sign(phi[i]) == -sign][:1]
    facts, sents = [], []

    p_lead = prob if decision == 1 else 1 - prob
    sents.append(f"{TIERS[base][3]} is patient ka profile <b>{lead_cls}</b> hai (model ke hisaab se {CLASS_NAMES[1]} probability {pct(prob)}, "
                 f"threshold {threshold * 100:.0f}%).")
    facts.append({"type": "headline", "decision": decision, "prob": prob})

    def phrase(i):
        f = FEATURES[i]; est = " (estimated value)" if f in filled else ""
        pct = ""
        if population is not None and f in population and f in UNIT:
            share = float((np.asarray(population[f]) < xs[i]).mean() * 100)
            pct = f", jo dataset ke {share:.0f}% patients se zyada hai" if share >= 50 else f", jo dataset ke {100 - share:.0f}% patients se kam hai"
        return f"{HUMAN[f]} = {value_text(f, xs[i])}{est}{pct} (<b>{abs(phi[i]) * 100:.1f} pts</b>)"

    if for_:
        sents.append(f"Sabse zyada asar: {'; '.join(phrase(i) for i in for_)} - ye {lead_cls} ki taraf dhakel rahe hain.")
        facts += [{"type": "for", "feature": FEATURES[i], "pts": float(abs(phi[i]) * 100), "dir": sign} for i in for_]
    if against and tier != "high":
        i = against[0]
        sents.append(f"Ulta asar: {phrase(i)} - ye {other_cls} ki taraf khinch raha hai.")
        facts.append({"type": "against", "feature": FEATURES[i], "pts": float(abs(phi[i]) * 100), "dir": -sign})

    if tier == "high":
        sents.append("Kai features ek hi taraf ishara kar rahe hain, isliye ye result stable lagta hai.")
    else:
        why = " ".join(r[0].upper() + r[1:] + "." for r in reasons)
        close = base in ("low", "toss-up")                 # margin sach me kam hai?
        line = {"moderate": "Result ek taraf jhuka hai, par pakka nahi.",
                "low": "Ye close call hai: thoda sa input badalne par result palat sakta hai." if close else
                       "Is result par bharosa kam hai; ise pakka mat maano.",
                "toss-up": "Ye lagbhag 50-50 hai; is result par zyada bharosa mat karo." if close else
                           "Is result par bharosa kaafi kam hai; ise pakka mat maano."}[tier]
        sents.append((line + " " + why).strip())

    cf = None
    if model is not None and rules is not None and x is not None and tier != "high":
        cf = counterfactual(model, x, rules, threshold)
        if cf:
            sents.append(f"Model ke hisaab se sirf {HUMAN[cf['feature']]} {value_text(cf['feature'], cf['from'])} se "
                         f"{value_text(cf['feature'], cf['to'])} hota to prediction {other_cls} ho jaati ({CLASS_NAMES[1]} probability {pct(cf['p_after'])}).")
        else:
            sents.append("In 4 numbers (cholesterol, BP, heart rate, oldpeak) me se akele ek badalne se prediction nahi palatti.")
    sents.append("Ye model ke data-patterns hain, doctor ki salah ya karan (cause) nahi.")
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
