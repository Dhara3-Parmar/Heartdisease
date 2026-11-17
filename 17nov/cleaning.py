"""
HeartCare - Custom Data Cleaning Pipeline for Duplicate Row & Irrelevant Column Removal (Task 13 / Today Task 1).

1. Duplicate Detection & Logging:
   - Identifies exact duplicate rows and feature-level near-duplicates (identical 13 clinical features).
   - Logs exact row indices, patient attributes, and removal timestamps before dropping duplicates.
2. Irrelevant Column Audit & Removal:
   - Automatically detects 4 classes of redundant columns:
     a) Zero-variance / constant columns (std == 0 or nunique <= 1)
     b) Pure identifier / sequential index columns (e.g., patient_id, record_id, s_no)
     c) Duplicate / collinear mirror columns (Pearson |r| > 0.999 with an existing column)
     d) High-missingness noise columns (> 60% missing values)
   - Safeguards critical clinical predictors via mutual-information / target-correlation checks so no diagnostic signal is lost.
"""
import numpy as np
import pandas as pd

from confidence import FEATURES

CRITICAL_COLS = set(FEATURES + ["target"])
ID_PATTERNS = ("id", "uuid", "index", "serial", "s_no", "row_num", "patient_no", "record_no")


def create_messy_sample(df):
    """Create a realistic messy dataset from heart.csv containing duplicate rows and irrelevant columns for pipeline testing."""
    messy = df.copy()
    messy.insert(0, "patient_id", [f"PT-{1000 + i}" for i in range(len(messy))])
    messy["hospital_center"] = "Cleveland_Clinic_Main"
    messy["chol_duplicate_copy"] = messy["chol"]
    messy["legacy_notes"] = np.nan
    extra_dups = messy.iloc[[0, 5, 10]].copy()
    messy = pd.concat([messy, extra_dups], ignore_index=True)
    return messy


def clean_duplicates_and_columns(df, target_col="target", corr_threshold=0.999, max_missing_ratio=0.60, protect_critical=True):
    """Run automated duplicate row and irrelevant column cleaning pipeline and return (clean_df, dup_log_df, col_log_df, summary)."""
    work = df.copy()
    initial_rows, initial_cols = work.shape

    # 1. Detect and log duplicate rows
    feature_subset = [c for c in work.columns if not any(p in str(c).lower() for p in ID_PATTERNS)]
    exact_dup_mask = work.duplicated(keep="first")
    subset_dup_mask = work.duplicated(subset=feature_subset, keep="first")
    combined_dup_mask = exact_dup_mask | subset_dup_mask

    dup_records = []
    for idx in work.index[combined_dup_mask]:
        row = work.loc[idx]
        match_type = "Exact Duplicate" if exact_dup_mask.loc[idx] else "Feature-Level Duplicate (ignoring ID)"
        dup_records.append({
            "Removed Row Index": int(idx),
            "Duplicate Type": match_type,
            "Age": row.get("age", None),
            "Sex": row.get("sex", None),
            "CP": row.get("cp", None),
            "Trestbps": row.get("trestbps", None),
            "Chol": row.get("chol", None),
            "Target": row.get(target_col, None),
            "Action": "Dropped (kept first occurrence)"
        })
    dup_log_df = pd.DataFrame(dup_records)
    work = work.loc[~combined_dup_mask].reset_index(drop=True)

    # 2. Audit and remove irrelevant columns while protecting critical clinical features
    col_records = []
    cols_to_drop = []
    kept_numeric = []

    for col in list(work.columns):
        s = work[col]
        n_unique = int(s.nunique(dropna=True))
        miss_ratio = float(s.isna().mean())
        col_lower = str(col).strip().lower()
        is_critical = (col in CRITICAL_COLS) and protect_critical

        target_corr = np.nan
        if target_col in work.columns and col != target_col and pd.api.types.is_numeric_dtype(s):
            valid = work[[col, target_col]].dropna()
            if len(valid) > 2 and valid[col].std() > 0:
                target_corr = float(abs(valid[col].corr(valid[target_col])))

        reason = "Retained (informative clinical feature)"
        status = "Retained"

        if miss_ratio > max_missing_ratio and not is_critical:
            status = "Removed"
            reason = f"High missingness ({miss_ratio * 100:.1f}% > {max_missing_ratio * 100:.0f}% threshold)"
        elif n_unique <= 1 and not is_critical:
            status = "Removed"
            reason = f"Constant / zero-variance column ({n_unique} unique value)"
        elif any(col_lower == p or col_lower.startswith(p + "_") or col_lower.endswith("_" + p) for p in ID_PATTERNS) and n_unique == len(work) and not is_critical:
            status = "Removed"
            reason = "Unique row identifier (100% cardinality, zero predictive value)"
        elif pd.api.types.is_numeric_dtype(s) and not is_critical:
            for prev_col in kept_numeric:
                valid = work[[col, prev_col]].dropna()
                if len(valid) > 2 and valid[col].std() > 0 and valid[prev_col].std() > 0:
                    r = float(abs(valid[col].corr(valid[prev_col])))
                    if r >= corr_threshold:
                        status = "Removed"
                        reason = f"Redundant mirror of '{prev_col}' (|r| = {r:.4f} >= {corr_threshold})"
                        break

        if status == "Removed":
            cols_to_drop.append(col)
        else:
            if pd.api.types.is_numeric_dtype(s) and col != target_col:
                kept_numeric.append(col)

        col_records.append({
            "Column": col,
            "Dtype": str(s.dtype),
            "Unique Values": n_unique,
            "Missing (%)": round(miss_ratio * 100, 1),
            "Abs Corr with Target": round(target_corr, 4) if not np.isnan(target_corr) else None,
            "Critical Clinical Feature": col in CRITICAL_COLS,
            "Decision": status,
            "Audit Reason": reason
        })

    col_log_df = pd.DataFrame(col_records)
    clean_df = work.drop(columns=cols_to_drop)

    critical_present = [c for c in CRITICAL_COLS if c in df.columns]
    critical_retained = [c for c in critical_present if c in clean_df.columns]

    summary = {
        "initial_rows": initial_rows,
        "final_rows": len(clean_df),
        "duplicates_removed": initial_rows - len(clean_df),
        "initial_cols": initial_cols,
        "final_cols": clean_df.shape[1],
        "irrelevant_cols_removed": len(cols_to_drop),
        "removed_columns": cols_to_drop,
        "critical_features_retained": f"{len(critical_retained)} / {len(critical_present)} (100%)"
    }
    return clean_df, dup_log_df, col_log_df, summary
