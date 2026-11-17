"""
HeartCare - Dynamic Column Name Standardizer & Data Type Parser (Task 14 / Today Task 2).

1. Column Name Standardization (standardize_columns):
   - Strips leading/trailing whitespace, converts CamelCase / spaces / hyphens / punctuation to snake_case.
   - Resolves clinical aliases (e.g., "Resting BP (mmHg)" -> "trestbps", "Serum Cholesterol" -> "chol") and fuzzy-matches typos.
2. Dynamic Data Type Parser (parse_and_cast_dtypes):
   - Automatically strips embedded measurement units ("145 mmHg", "233 mg/dL", "63 yrs", "150 bpm").
   - Normalizes comma decimals ("2,3" -> 2.3), boolean strings ("True"/"Male"/"Yes"), and missing tokens.
   - Casts integer-valued clinical codes and whole numbers (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `slope`, `ca`, `thal`, `target`) to `int64` (or `Int64` if nullable) and continuous decimal features (`oldpeak`) to `float64`.
"""
import difflib
import re
import numpy as np
import pandas as pd

from confidence import FEATURES

CANONICAL_COLS = FEATURES + ["target"]

COLUMN_ALIASES = {
    "patient_age": "age", "age_years": "age", "age_in_years": "age",
    "gender": "sex", "patient_sex": "sex",
    "chest_pain": "cp", "chest_pain_type": "cp", "cp_type": "cp",
    "bp": "trestbps", "resting_bp": "trestbps", "resting_blood_pressure": "trestbps", "trest_bps": "trestbps", "blood_pressure": "trestbps",
    "cholesterol": "chol", "serum_chol": "chol", "serum_cholesterol": "chol", "chol_mg_dl": "chol",
    "fasting_blood_sugar": "fbs", "fasting_sugar": "fbs",
    "resting_ecg": "restecg", "rest_ecg": "restecg", "ecg": "restecg",
    "max_heart_rate": "thalach", "max_hr": "thalach", "heart_rate": "thalach", "thalach_bpm": "thalach",
    "exercise_angina": "exang", "ex_angina": "exang",
    "st_depression": "oldpeak", "old_peak": "oldpeak",
    "st_slope": "slope",
    "major_vessels": "ca", "num_vessels": "ca", "vessels": "ca",
    "thalassemia": "thal", "thal_test": "thal",
    "heart_disease": "target", "diagnosis": "target", "label": "target", "outcome": "target"
}

TEXT_CODE_MAP = {
    "male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0,
    "yes": 1.0, "y": 1.0, "true": 1.0, "positive": 1.0, "high": 1.0,
    "no": 0.0, "n": 0.0, "false": 0.0, "negative": 0.0, "low": 0.0
}

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "unknown", "not known"}
INT_COLS = {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "slope", "ca", "thal", "target"}
FLOAT_COLS = {"oldpeak"}
UNIT_REGEX = re.compile(r"^([-+]?\d*[.,]?\d+)\s*(?:yrs?|years?|mm\s*hg|mmhg|mg\s*/\s*dl|mgdl|bpm|%|pts?)?$", re.IGNORECASE)


def to_snake_case(name):
    s = str(name).strip()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return re.sub(r"_+", "_", s)


def standardize_columns(df):
    """Standardize raw DataFrame column names to canonical snake_case names and return (renamed_df, mapping_log_df)."""
    new_cols = []
    mapping_rows = []
    for raw_col in df.columns:
        snake = to_snake_case(raw_col)
        if snake in CANONICAL_COLS:
            mapped, method = snake, "Direct / Case-Normalized Match"
        elif snake in COLUMN_ALIASES:
            mapped, method = COLUMN_ALIASES[snake], "Clinical Alias Lookup"
        else:
            matches = difflib.get_close_matches(snake, CANONICAL_COLS + list(COLUMN_ALIASES.keys()), n=1, cutoff=0.72)
            if matches:
                m = matches[0]
                mapped = COLUMN_ALIASES.get(m, m)
                method = f"Fuzzy Typo Match ('{snake}' -> '{mapped}')"
            else:
                mapped, method = snake, "Standardized snake_case"
        new_cols.append(mapped)
        mapping_rows.append({
            "Original Column Name": str(raw_col),
            "Snake Case Intermediate": snake,
            "Standardized Column": mapped,
            "Resolution Method": method
        })
    out = df.copy()
    out.columns = new_cols
    return out, pd.DataFrame(mapping_rows)


def _parse_cell(val):
    """Parse a single messy cell into (float_or_nan, was_repaired, had_error)."""
    if val is None or val is pd.NA:
        return np.nan, False, False
    if isinstance(val, (bool, np.bool_)):
        return float(int(val)), True, False
    if isinstance(val, (int, float, np.integer, np.floating)):
        if np.isnan(val) or not np.isfinite(val):
            return np.nan, False, False
        return float(val), False, False
    s = str(val).strip()
    low = s.lower()
    if low in MISSING_TOKENS:
        return np.nan, True, False
    if low in TEXT_CODE_MAP:
        return TEXT_CODE_MAP[low], True, False
    m = UNIT_REGEX.match(s)
    if m:
        num_str = m.group(1).replace(",", ".")
        try:
            repaired = (num_str != s)
            return float(num_str), repaired, False
        except ValueError:
            return np.nan, False, True
    try:
        return float(s.replace(",", ".")), ("," in s), False
    except ValueError:
        return np.nan, False, True


def parse_and_cast_dtypes(df):
    """Dynamically parse messy string/mixed columns and cast to canonical numeric dtypes."""
    std_df, col_map_df = standardize_columns(df)
    out = pd.DataFrame(index=std_df.index)
    dtype_rows = []

    for col in std_df.columns:
        orig_dtype = str(std_df[col].dtype)
        repaired_count = 0
        error_count = 0
        parsed_vals = []

        for v in std_df[col]:
            num, rep, err = _parse_cell(v)
            parsed_vals.append(num)
            repaired_count += int(rep)
            error_count += int(err)

        arr = np.array(parsed_vals, dtype=float)
        has_na = bool(np.isnan(arr).any())

        if col in INT_COLS or (col not in FLOAT_COLS and not has_na and np.all(np.mod(arr[~np.isnan(arr)], 1) == 0)):
            if not has_na:
                out[col] = np.round(arr).astype(np.int64)
            else:
                out[col] = pd.Series(np.round(arr), index=std_df.index).astype("Int64")
            target_kind = "Integer (int64)" if not has_na else "Nullable Integer (Int64)"
        else:
            out[col] = arr.astype(np.float64)
            target_kind = "Continuous Float (float64)"

        dtype_rows.append({
            "Column": col,
            "Original Dtype": orig_dtype,
            "Final Dtype": str(out[col].dtype),
            "Schema Category": target_kind,
            "Formatted Cells Repaired": repaired_count,
            "Unparseable Errors": error_count
        })

    return out, col_map_df, pd.DataFrame(dtype_rows)


def create_unformatted_sample(df, n_rows=25):
    """Create a messy formatting test DataFrame from heart.csv with inconsistent headers, units, comma decimals, and text booleans."""
    sub = df.head(n_rows).copy().astype(object)
    sub.loc[0, "age"] = "63 yrs"
    sub.loc[1, "trestbps"] = "130 mmHg"
    sub.loc[2, "chol"] = "204 mg/dL"
    sub.loc[3, "thalach"] = "178 bpm"
    sub.loc[4, "oldpeak"] = "0,6"
    sub.loc[5, "sex"] = "Male"
    sub.loc[6, "sex"] = "Female"
    sub.loc[7, "fbs"] = "False"
    sub.loc[8, "exang"] = "No"
    sub.columns = [
        "Patient Age (Yrs)", "Gender", "Chest Pain Type", "Resting BP (mmHg)",
        "Serum Cholesterol", "Fasting Blood Sugar", "Resting-ECG",
        "Max Heart Rate", "Exercise Angina", "Old_Peak", "ST Slope",
        "Major Vessels", "Thalassemia", "Heart Disease"
    ]
    return sub
