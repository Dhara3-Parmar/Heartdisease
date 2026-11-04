"""
HeartCare - robust input validation framework (detect -> correct -> explain -> suggest).

Layers:
  1. parse_quick(text)         : "age 63, chol 5.2 mmol, bp=130" jaisi free text ko record me todta hai (naam ke typo pakadta hai)
  2. diagnose(record, rules)   : har field ke liye problem dhundta hai. Teen tarah ke result:
        fixed   - safe auto-correction (jaise "1,5" -> 1.5, "63 yrs" -> 63, "5.2 mmol/L" -> 201 mg/dl)
        error   - prediction roki jaati hai; message + (agar ho sake to) suggested value
        warning - prediction chalti hai par user ko batate hain
  3. check_record(record, rules): diagnose + input.validate_input (purana validator) = final {ok, x, errors, warnings, filled, issues}
Rule: jo cheez galat ho use chupchap kabhi "theek" nahi karte; jo 100% safe hai wahi auto-fix hota hai, baaki par user ki haan chahiye.
"""
import difflib
import math
import re

import numpy as np

from input import FEATURES, MAX_MISSING, validate_input

ALIASES = {"bp": "trestbps", "blood_pressure": "trestbps", "resting_bp": "trestbps", "cholesterol": "chol",
           "heart_rate": "thalach", "max_hr": "thalach", "maxhr": "thalach", "st_depression": "oldpeak",
           "vessels": "ca", "chest_pain": "cp", "angina": "exang", "sugar": "fbs", "ecg": "restecg", "gender": "sex"}
MMOL_TO_MGDL = 38.67                      # cholesterol: 1 mmol/L = 38.67 mg/dl
UNIT_WORDS = {"yrs", "yr", "years", "year", "y", "mmhg", "mg/dl", "mgdl", "bpm", "mm", "hg", "%"}
EXAMPLE = {"age": "55", "trestbps": "130", "chol": "240", "thalach": "150", "oldpeak": "1.0"}


def issue(field, level, message, fix=None, fix_label=None, value=None):
    """fix = suggested value (ya naam-typo me suggested field ka naam); value = user ne jo likha tha (naam-typo ke liye)."""
    return {"field": field, "level": level, "message": message, "fix": fix, "fix_label": fix_label, "value": value}


def _norm(name):
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def match_field(name):
    """Field ka naam pehchano. Return (field ya None, suggestions list)."""
    n = _norm(name)
    if n in FEATURES:
        return n, []
    if n in ALIASES:
        return ALIASES[n], []
    close = difflib.get_close_matches(n, FEATURES + list(ALIASES), n=3, cutoff=0.6)
    close = [ALIASES.get(c, c) for c in close]
    close = list(dict.fromkeys(close))
    return None, close


def parse_quick(text):
    """Free text -> (record, issues). Separator: ; ya naya line ya ', ' ke baad naya naam (1,5 jaisa comma-decimal bacha rehta hai)."""
    record, issues = {}, []
    parts = re.split(r"[;\n]|,\s*(?=[A-Za-z])", str(text or ""))
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z_ \-]*?)\s*(?:[=:]\s*|\s+(?=[-+\d.,])|$)(.*)$", part)
        if m and match_field(m.group(1))[0] is None and " " in m.group(1).strip():       # "sex male": pehla shabd naam, baaki value
            first, rest = part.split(None, 1)
            m = re.match(r"^(.*)$", first), None
            m = type("M", (), {"group": lambda self, i, a=first, b=rest.lstrip("=: "): a if i == 1 else b})()
        if not m:
            issues.append(issue(None, "error", f"'{part}' samajh nahi aaya. Aise likho: age 63, chol 240"))
            continue
        field, close = match_field(m.group(1))
        if field is None:
            hint = f" Kya aap '{close[0]}' kehna chahte the?" if close else f" Fields: {', '.join(FEATURES)}"
            issues.append(issue(None, "error", f"'{m.group(1).strip()}' naam pehchana nahi gaya.{hint}",
                                fix=close[0] if close else None, fix_label=f"'{close[0]}' maano" if close else None,
                                value=m.group(2).strip()))
            continue
        if _norm(m.group(1)) != field:
            issues.append(issue(field, "fixed", f"'{m.group(1).strip()}' ko '{field}' samjha."))
        if field in record:
            issues.append(issue(field, "warning", f"{field} do baar likha hai; aakhri value li gayi."))
        record[field] = m.group(2).strip()
    return record, issues


_NUM = re.compile(r"^([-+]?\d*[.,]?\d+)\s*([A-Za-z/%]*[A-Za-z%])?$")


def _to_number(field, raw, rules):
    """String -> (number ya None, issues). Comma decimal, unit suffix, mmol/L conversion yahan hota hai."""
    out = []
    s = str(raw).strip()
    m = _NUM.match(s.replace(" ", "")) if re.match(r"^[-+]?\.?\d", s) else None
    if m is None:
        r = rules[field]
        ex = EXAMPLE.get(field, "")
        rng = f"{r['min']:g} se {r['max']:g}" if r["kind"] == "number" else "allowed: " + "/".join(str(int(a)) for a in r["allowed"])
        out.append(issue(field, "error", f"{field}: '{s}' number nahi hai. Sirf number likho ({rng})" + (f", jaise {ex}." if ex else "."),
                         fix=float(r["fill"]), fix_label=f"typical value {r['fill']:g} lo"))
        return None, out
    num_txt, unit = m.group(1), (m.group(2) or "").lower()
    if "," in num_txt:
        out.append(issue(field, "fixed", f"{field}: '{num_txt}' ko {num_txt.replace(',', '.')} samjha (comma = decimal)."))
        num_txt = num_txt.replace(",", ".")
    value = float(num_txt)
    if unit:
        if unit in ("mmol/l", "mmol", "mmoll") and field == "chol":
            out.append(issue(field, "fixed", f"chol: {value:g} mmol/L ko {value * MMOL_TO_MGDL:.0f} mg/dl me badla (x{MMOL_TO_MGDL})."))
            value = value * MMOL_TO_MGDL
        elif unit in UNIT_WORDS:
            out.append(issue(field, "fixed", f"{field}: unit '{unit}' hata diya, value {value:g} li."))
        else:
            out.append(issue(field, "error", f"{field}: unit '{unit}' pehchana nahi gaya. Sirf number likho."))
            return None, out
    return value, out


def _suggest_number(field, v, r):
    """Hard limit ke bahar number ke liye sabse samajhdar suggestion."""
    if field == "chol" and 2 <= v < 40:                                    # mmol/L jaisa lagta hai
        fixed = v * MMOL_TO_MGDL
        if r["hard_lo"] <= fixed <= r["hard_hi"]:
            return issue(field, "error", f"chol: {v:g} bahut kam hai (mg/dl me). Kya ye mmol/L me hai? To {fixed:.0f} mg/dl banta hai.",
                         fix=round(fixed), fix_label=f"{fixed:.0f} mg/dl lo")
    if field == "age" and v > r["hard_hi"] and v / 12 >= r["hard_lo"] and v / 12 <= r["hard_hi"]:
        return issue(field, "error", f"age: {v:g} saal impossible hai. Kya ye mahine hain? To {v / 12:.0f} saal banta hai.",
                     fix=round(v / 12), fix_label=f"{v / 12:.0f} saal lo")
    if v < 0 and r["hard_lo"] <= abs(v) <= r["hard_hi"]:
        return issue(field, "error", f"{field}: {v:g} negative nahi ho sakta. Kya {abs(v):g} matlab tha?", fix=abs(v), fix_label=f"{abs(v):g} lo")
    near = min(max(v, r["min"]), r["max"])                  # data ki range ke andar sabse paas ki value
    return issue(field, "error", f"{field}: {v:g} impossible hai (limit {r['hard_lo']:g} se {r['hard_hi']:g}). Aam range ({r['min']:g} se {r['max']:g}) me sabse paas ki value {near:g} hai.",
                 fix=round(near, 2), fix_label=f"{near:g} lo")


def diagnose(record, rules, model=None, partial=False):
    """Return {'clean': record (auto-fix ke baad), 'issues': [...], 'blocked': bool}.
    partial=True: sirf diye gaye fields jaanchte hain (quick entry), missing-count ka rule nahi lagta."""
    issues, clean, missing = [], {}, []
    rec = dict(record) if isinstance(record, dict) else {}
    for f in FEATURES:
        raw = rec.get(f)
        r = rules[f]
        if raw is None or (isinstance(raw, str) and raw.strip() == "") or (isinstance(raw, float) and math.isnan(raw)):
            clean[f] = None; missing.append(f); continue
        if isinstance(raw, bool):
            raw = int(raw)
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in ("true", "false"):
                raw = 1 if low == "true" else 0
            else:
                raw, extra = _to_number(f, raw, rules); issues += extra
                if raw is None:
                    clean[f] = None; continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            issues.append(issue(f, "error", f"{f}: '{raw}' number nahi hai.", fix=float(r["fill"]), fix_label=f"typical value {r['fill']:g} lo"))
            clean[f] = None; continue
        if not math.isfinite(v):
            issues.append(issue(f, "error", f"{f}: infinity/NaN allowed nahi.", fix=float(r["fill"]), fix_label=f"typical value {r['fill']:g} lo"))
            clean[f] = None; continue
        if r["kind"] == "category":
            allowed = [float(a) for a in r["allowed"]]
            if v not in allowed:
                order = sorted(allowed, key=lambda a: (abs(a - v), a))
                alt = " ya ".join(f"{int(a)}" for a in order[:2]) if abs(order[0] - v) == abs(order[1] - v) else f"{int(order[0])}"
                issues.append(issue(f, "error", f"{f}: {v:g} allowed nahi (allowed: {'/'.join(str(int(a)) for a in allowed)}). Paas ki value: {alt}.",
                                    fix=int(order[0]), fix_label=f"{int(order[0])} lo"))
                clean[f] = None; continue
        else:
            if not (r["hard_lo"] <= v <= r["hard_hi"]):
                issues.append(_suggest_number(f, v, r)); clean[f] = None; continue
            if not (r["min"] <= v <= r["max"]):
                issues.append(issue(f, "warning", f"{f}: {v:g} training range ({r['min']:g} se {r['max']:g}) ke bahar hai; prediction kam bharosemand ho sakti hai."))
        clean[f] = v

    # fields ke beech ki jaanch (sirf chetavni)
    a, hr = clean.get("age"), clean.get("thalach")
    if a is not None and hr is not None and hr > (220 - a) + 15:
        issues.append(issue("thalach", "warning", f"thalach {hr:g} is umar (age {a:g}) ke aam max heart rate (~{220 - a:g}) se kaafi upar hai. Value dobara check kar lo."))

    # bahut zyada missing: sabse kaam ke fields batao
    if not partial and len(missing) > MAX_MISSING:
        pri = _priority(model)
        ask = [f for f in pri if f in missing][: len(missing) - MAX_MISSING]
        issues.append(issue(None, "error", f"{len(missing)} fields khali hain (max {MAX_MISSING}). Kam se kam {len(missing) - MAX_MISSING} aur bharo - "
                            f"sabse zyada madad karne wale: {', '.join(ask)}."))
    return {"clean": clean, "issues": issues, "blocked": any(i["level"] == "error" for i in issues)}


def _priority(model):
    if model is not None and hasattr(model, "feature_importances_"):
        return [FEATURES[i] for i in np.argsort(-model.feature_importances_)]
    return list(FEATURES)


def check_record(record, rules, model=None):
    """Poora pipeline. Return input.validate_input jaisa dict + 'issues'."""
    d = diagnose(record, rules, model)
    if d["blocked"]:
        errs = [i["message"] for i in d["issues"] if i["level"] == "error"]
        warns = [i["message"] for i in d["issues"] if i["level"] != "error"]
        return {"ok": False, "x": None, "errors": errs, "warnings": warns, "filled": [], "issues": d["issues"]}
    res = validate_input(d["clean"], rules)
    notes = [i["message"] for i in d["issues"] if i["level"] in ("warning", "fixed")]
    res["warnings"] = notes + [w for w in res["warnings"] if w not in notes]
    res["issues"] = d["issues"]
    return res


def apply_fix(record, iss):
    """Suggestion maanne par record badalta hai. Naam-typo me value naye field par jaati hai, warna field ki value badalti hai."""
    out = dict(record)
    if iss["field"] is None and iss.get("value") is not None:      # naam-typo: 'choll' -> chol
        out[iss["fix"]] = iss["value"]
    elif iss["field"] in FEATURES:
        out[iss["field"]] = iss["fix"]
    return out
