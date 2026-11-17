"""
HeartCare - user input handling for prediction.

1. Dynamic validation : rules (allowed categories, number ranges) built automatically from heart.csv.
2. Edge cases         : missing, text, out-of-range, unknown category, inf/NaN, extra keys, wrong length.
3. Real-time speed    : single predict_proba pass, LRU cache for repeated inputs, vectorized batch validation.
"""
import numpy as np
import pandas as pd
from functools import lru_cache

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
MISSING_WORDS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "unknown", "not known"}
MAX_MISSING = 3


def build_rules(df):
    """Build validation rules from training data: <= 5 unique values = category, otherwise = number."""
    rules = {}
    for f in FEATURES:
        col = df[f].astype(float).dropna()
        vals = np.sort(col.unique())
        r = {"min": float(col.min()), "max": float(col.max())}
        if len(vals) <= 5:
            r.update(kind="category", allowed=vals.tolist(), fill=float(col.mode()[0]))
        else:
            span = col.max() - col.min()
            lo, hi = col.min() - 0.25 * span, col.max() + 0.25 * span
            if col.min() >= 0:
                lo = max(lo, 0.0)
            r.update(kind="number", hard_lo=float(lo), hard_hi=float(hi), fill=float(col.median()))
        rules[f] = r
    return rules


def load_rules(csv_path="heart.csv"):
    return build_rules(pd.read_csv(csv_path).drop_duplicates().dropna())


def is_missing(v):
    if v is None or v is pd.NA:
        return True
    if isinstance(v, str):
        return v.strip().lower() in MISSING_WORDS
    return isinstance(v, (float, np.floating)) and np.isnan(v)


def to_number(v):
    if isinstance(v, (bool, int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().lower().replace(",", ".")
        return {"true": 1.0, "false": 0.0}.get(t) if t in ("true", "false") else float(t)
    raise TypeError("Not a valid number")


def validate_input(record, rules):
    """record = dict or list of 13 values. Returns: {ok, x, errors, warnings, filled}"""
    out = {"ok": False, "x": None, "errors": [], "warnings": [], "filled": []}
    if isinstance(record, (list, tuple, np.ndarray)):
        if len(record) != len(FEATURES):
            out["errors"].append(f"Expected {len(FEATURES)} values, got {len(record)}")
            return out
        record = dict(zip(FEATURES, record))
    if not isinstance(record, dict):
        out["errors"].append("Input must be a dictionary or list")
        return out

    rec = {str(k).strip().lower().replace(" ", "_"): v for k, v in record.items()}
    extra = [k for k in rec if k not in rules]
    if extra:
        out["warnings"].append(f"Ignored extra fields: {extra}")

    x = np.zeros(len(FEATURES))
    for i, f in enumerate(FEATURES):
        v, r = rec.get(f), rules[f]
        if is_missing(v):
            x[i] = r["fill"]; out["filled"].append(f); continue
        try:
            val = to_number(v)
        except (ValueError, TypeError):
            out["errors"].append(f"{f}: '{v}' is not a valid number"); continue
        if not np.isfinite(val):
            out["errors"].append(f"{f}: inf/NaN is not allowed"); continue
        if r["kind"] == "category":
            if val not in r["allowed"]:
                out["errors"].append(f"{f}: {val:g} is not allowed (allowed: {[int(a) for a in r['allowed']]})")
        elif not r["hard_lo"] <= val <= r["hard_hi"]:
            out["errors"].append(f"{f}: {val:g} is out of physiological bounds (limit {r['hard_lo']:.1f} to {r['hard_hi']:.1f})")
        elif not r["min"] <= val <= r["max"]:
            out["warnings"].append(f"{f}: {val:g} is outside training range [{r['min']:g}, {r['max']:g}]; prediction may be less reliable")
        x[i] = val

    if len(out["filled"]) > MAX_MISSING:
        out["errors"].append(f"{len(out['filled'])} fields are missing (maximum {MAX_MISSING} allowed)")
    elif out["filled"]:
        out["warnings"].append(f"Missing fields filled with median/mode: {out['filled']}")
    if not out["errors"]:
        out["ok"], out["x"] = True, x
    return out


def predict_result(model, x):
    """Single predict_proba pass for both class and probability."""
    probabilities = model.predict_proba(pd.DataFrame([x], columns=FEATURES))[0]
    prediction = int(model.classes_[np.argmax(probabilities)])
    return prediction, probabilities


def make_predictor(model, size=1024):
    """LRU cached predictor so repeated inputs skip model execution."""
    @lru_cache(maxsize=size)
    def cached(key):
        return predict_result(model, list(key))
    return lambda x: cached(tuple(float(v) for v in x))


def _to_numbers(s):
    """Series -> (numbers, missing_mask). Non-numeric text becomes NaN and is flagged as invalid rather than missing."""
    if pd.api.types.is_numeric_dtype(s):
        num = s.astype(float).to_numpy()
        return num, np.isnan(num)
    txt = s.astype(str).str.strip().str.lower()
    miss = (s.isna() | txt.isin(MISSING_WORDS)).to_numpy()
    num = pd.to_numeric(txt.str.replace(",", ".", regex=False).replace({"true": "1", "false": "0"}), errors="coerce").to_numpy(dtype=float)
    return num, miss


def validate_batch(raw, rules):
    """Vectorized validation across all rows of a DataFrame. Returns: X, ok(bool array), reason(Series)"""
    raw = raw.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    n = len(raw)
    X, bad, n_miss = np.zeros((n, len(FEATURES))), np.zeros(n, bool), np.zeros(n, int)
    reason = pd.Series([""] * n, index=raw.index, dtype=object)
    for i, f in enumerate(FEATURES):
        r = rules[f]
        if f in raw:
            num, miss = _to_numbers(raw[f])
        else:
            num, miss = np.full(n, np.nan), np.ones(n, bool)
        bad_f = ~miss & ~np.isfinite(num)
        ok_num = ~miss & np.isfinite(num)
        if r["kind"] == "category":
            bad_f |= ok_num & ~np.isin(num, r["allowed"])
        else:
            bad_f |= ok_num & ((num < r["hard_lo"]) | (num > r["hard_hi"]))
        reason = reason + np.where(bad_f, f + " invalid; ", "")
        bad |= bad_f; n_miss += miss
        X[:, i] = np.where(miss | bad_f, r["fill"], num)
    too_many = n_miss > MAX_MISSING
    reason = reason + np.where(too_many, "too many missing fields; ", "")
    return X, ~(bad | too_many), reason


def predict_batch(model, rules, raw):
    """Single vectorized predict_proba call over all valid rows in a CSV/DataFrame."""
    X, ok, reason = validate_batch(raw, rules)
    out = pd.DataFrame({"status": np.where(ok, "ok", "rejected"), "reason": reason.str.strip("; ").to_numpy(),
                        "prediction": np.nan, "probability_%": np.nan}, index=raw.index)
    if ok.any():
        p = model.predict_proba(pd.DataFrame(X[ok], columns=FEATURES))
        out.loc[ok, "prediction"] = model.classes_[p.argmax(axis=1)]
        out.loc[ok, "probability_%"] = (p.max(axis=1) * 100).round(2)
    return out
