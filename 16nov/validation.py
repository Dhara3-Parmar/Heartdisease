"""
HeartCare - robust input validation framework (detect -> correct -> explain -> suggest).

Layers:
  1. parse_quick(text)          : parses free-text quick entry (e.g. "age 63, chol 5.2 mmol, bp=130") and detects field name typos
  2. diagnose(record, rules)    : inspects each field and categorizes findings into:
        fixed   - safe automatic conversions (e.g. "1,5" -> 1.5, "63 yrs" -> 63, "5.2 mmol/L" -> 201 mg/dL)
        error   - blocks prediction with a contextual message and intelligent suggested value
        warning - allows prediction but alerts the user (e.g. value outside training range)
  3. check_record(record, rules): combines diagnose() + input.validate_input()
"""
import difflib
import math
import re

import numpy as np

from input import FEATURES, MAX_MISSING, validate_input

ALIASES = {"bp": "trestbps", "blood_pressure": "trestbps", "resting_bp": "trestbps", "cholesterol": "chol",
           "heart_rate": "thalach", "max_hr": "thalach", "maxhr": "thalach", "st_depression": "oldpeak",
           "vessels": "ca", "chest_pain": "cp", "angina": "exang", "sugar": "fbs", "ecg": "restecg", "gender": "sex"}
MMOL_TO_MGDL = 38.67
UNIT_WORDS = {"yrs", "yr", "years", "year", "y", "mmhg", "mg/dl", "mgdl", "bpm", "mm", "hg", "%"}
EXAMPLE = {"age": "55", "trestbps": "130", "chol": "240", "thalach": "150", "oldpeak": "1.0"}


def issue(field, level, message, fix=None, fix_label=None, value=None):
    return {"field": field, "level": level, "message": message, "fix": fix, "fix_label": fix_label, "value": value}


def _norm(name):
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def match_field(name):
    """Resolve field name or suggest close matches."""
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
    """Parse free-text quick entry into (record, issues)."""
    record, issues = {}, []
    parts = re.split(r"[;\n]|,\s*(?=[A-Za-z])", str(text or ""))
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z_ \-]*?)\s*(?:[=:]\s*|\s+(?=[-+d\d.,])|$)(.*)$", part)
        if m and match_field(m.group(1))[0] is None and " " in m.group(1).strip():
            first, rest = part.split(None, 1)
            m = type("M", (), {"group": lambda self, i, a=first, b=rest.lstrip("=: "): a if i == 1 else b})()
        if not m:
            issues.append(issue(None, "error", f"Could not parse '{part}'. Use format: age 63, chol 240"))
            continue
        field, close = match_field(m.group(1))
        if field is None:
            hint = f" Did you mean '{close[0]}'?" if close else f" Valid fields: {', '.join(FEATURES)}"
            issues.append(issue(None, "error", f"Unrecognized field '{m.group(1).strip()}'.{hint}",
                                fix=close[0] if close else None, fix_label=f"Use '{close[0]}'" if close else None,
                                value=m.group(2).strip()))
            continue
        if _norm(m.group(1)) != field:
            issues.append(issue(field, "fixed", f"Mapped '{m.group(1).strip()}' to '{field}'."))
        if field in record:
            issues.append(issue(field, "warning", f"Field '{field}' was entered twice; using the last value."))
        record[field] = m.group(2).strip()
    return record, issues


_NUM = re.compile(r"^([-+]?\d*[.,]?\d+)\s*([A-Za-z/%]*[A-Za-z%])?$")


def _to_number(field, raw, rules):
    """Parse string into (number or None, issues) with comma-decimal and unit handling."""
    out = []
    s = str(raw).strip()
    m = _NUM.match(s.replace(" ", "")) if re.match(r"^[-+]?\.?\d", s) else None
    if m is None:
        r = rules[field]
        ex = EXAMPLE.get(field, "")
        rng = f"{r['min']:g} to {r['max']:g}" if r["kind"] == "number" else "allowed: " + "/".join(str(int(a)) for a in r["allowed"])
        out.append(issue(field, "error", f"{field}: '{s}' is not a valid number. Enter a number ({rng})" + (f", e.g. {ex}." if ex else "."),
                         fix=float(r["fill"]), fix_label=f"Use typical value {r['fill']:g}"))
        return None, out
    num_txt, unit = m.group(1), (m.group(2) or "").lower()
    if "," in num_txt:
        out.append(issue(field, "fixed", f"{field}: converted comma-decimal '{num_txt}' to {num_txt.replace(',', '.')}."))
        num_txt = num_txt.replace(",", ".")
    value = float(num_txt)
    if unit:
        if unit in ("mmol/l", "mmol", "mmoll") and field == "chol":
            out.append(issue(field, "fixed", f"chol: converted {value:g} mmol/L to {value * MMOL_TO_MGDL:.0f} mg/dL (x{MMOL_TO_MGDL})."))
            value = value * MMOL_TO_MGDL
        elif unit in UNIT_WORDS:
            out.append(issue(field, "fixed", f"{field}: stripped unit '{unit}', kept numeric value {value:g}."))
        else:
            out.append(issue(field, "error", f"{field}: unrecognized unit '{unit}'. Please enter a plain number."))
            return None, out
    return value, out


def _suggest_number(field, v, r):
    """Intelligent suggestion for out-of-bounds numerical inputs."""
    if field == "chol" and 2 <= v < 40:
        fixed = v * MMOL_TO_MGDL
        if r["hard_lo"] <= fixed <= r["hard_hi"]:
            return issue(field, "error", f"chol: {v:g} is too low for mg/dL. If entered in mmol/L, it equals {fixed:.0f} mg/dL.",
                         fix=round(fixed), fix_label=f"Use {fixed:.0f} mg/dL")
    if field == "age" and v > r["hard_hi"] and r["hard_lo"] <= v / 12 <= r["hard_hi"]:
        return issue(field, "error", f"age: {v:g} years is invalid. If entered in months, it equals {v / 12:.0f} years.",
                     fix=round(v / 12), fix_label=f"Use {v / 12:.0f} years")
    if v < 0 and r["hard_lo"] <= abs(v) <= r["hard_hi"]:
        return issue(field, "error", f"{field}: {v:g} cannot be negative. Did you mean {abs(v):g}?", fix=abs(v), fix_label=f"Use {abs(v):g}")
    near = min(max(v, r["min"]), r["max"])
    return issue(field, "error", f"{field}: {v:g} is out of bounds (limit {r['hard_lo']:g} to {r['hard_hi']:g}). Nearest typical value is {near:g}.",
                 fix=round(near, 2), fix_label=f"Use {near:g}")


def diagnose(record, rules, model=None, partial=False):
    """Diagnose input record and return {'clean': dict, 'issues': list, 'blocked': bool}."""
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
            issues.append(issue(f, "error", f"{f}: '{raw}' is not a valid number.", fix=float(r["fill"]), fix_label=f"Use typical value {r['fill']:g}"))
            clean[f] = None; continue
        if not math.isfinite(v):
            issues.append(issue(f, "error", f"{f}: infinity/NaN is not allowed.", fix=float(r["fill"]), fix_label=f"Use typical value {r['fill']:g}"))
            clean[f] = None; continue
        if r["kind"] == "category":
            allowed = [float(a) for a in r["allowed"]]
            if v not in allowed:
                order = sorted(allowed, key=lambda a: (abs(a - v), a))
                alt = " or ".join(f"{int(a)}" for a in order[:2]) if abs(order[0] - v) == abs(order[1] - v) else f"{int(order[0])}"
                issues.append(issue(f, "error", f"{f}: {v:g} is not allowed (allowed: {'/'.join(str(int(a)) for a in allowed)}). Nearest code: {alt}.",
                                    fix=int(order[0]), fix_label=f"Use {int(order[0])}"))
                clean[f] = None; continue
        else:
            if not (r["hard_lo"] <= v <= r["hard_hi"]):
                issues.append(_suggest_number(f, v, r)); clean[f] = None; continue
            if not (r["min"] <= v <= r["max"]):
                issues.append(issue(f, "warning", f"{f}: {v:g} is outside the training range ({r['min']:g} to {r['max']:g}); prediction may be less reliable."))
        clean[f] = v

    a, hr = clean.get("age"), clean.get("thalach")
    if a is not None and hr is not None and hr > (220 - a) + 15:
        issues.append(issue("thalach", "warning", f"thalach {hr:g} bpm exceeds the typical age-adjusted max heart rate (~{220 - a:g} bpm for age {a:g}). Please verify."))

    if not partial and len(missing) > MAX_MISSING:
        pri = _priority(model)
        ask = [f for f in pri if f in missing][: len(missing) - MAX_MISSING]
        issues.append(issue(None, "error", f"{len(missing)} fields are empty (maximum {MAX_MISSING} allowed). Please provide at least {len(missing) - MAX_MISSING} more — "
                            f"most informative features: {', '.join(ask)}."))
    return {"clean": clean, "issues": issues, "blocked": any(i["level"] == "error" for i in issues)}


def _priority(model):
    if model is not None and hasattr(model, "feature_importances_"):
        return [FEATURES[i] for i in np.argsort(-model.feature_importances_)]
    return list(FEATURES)


def check_record(record, rules, model=None):
    """Full validation pipeline returning {ok, x, errors, warnings, filled, issues}."""
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
    """Apply a suggested fix to the input record."""
    out = dict(record)
    if iss["field"] is None and iss.get("value") is not None:
        out[iss["fix"]] = iss["value"]
    elif iss["field"] in FEATURES:
        out[iss["field"]] = iss["fix"]
    return out
