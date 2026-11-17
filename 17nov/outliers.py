"""
HeartCare - Multi-Method Outlier Detection (IQR, Z-Score, DBSCAN) & Domain-Driven Treatment (Task 15 / Today Task 3).

1. Multi-Method Outlier Detection on Continuous Clinical Features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`):
   - Method A: Interquartile Range (IQR) Rule -> outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
   - Method B: Parametric Z-Score Rule        -> |Z| = |(x - mean) / std| > 3.0
   - Method C: Multivariate Density Clustering (DBSCAN) -> `sklearn.cluster.DBSCAN(eps=2.2, min_samples=5)` on standardized features (cluster label == -1)
2. Case-by-Case Domain-Specific Treatment Strategy:
   - `age`      : Retain as-is (29 to 77 years are completely normal adult cardiology ages; 0 outliers).
   - `trestbps` : Winsorize / Cap at 99th percentile (180 mmHg) — extreme hypertensive crisis readings (>180 mmHg, up to 200) distort linear distance metrics.
   - `chol`     : Log-transform (`np.log1p`) + cap extreme tail (>400 mg/dL, e.g. 564 mg/dL) — serum cholesterol is right-skewed (skew = 1.14).
   - `thalach`  : Winsorize / Cap lower tail at 1st percentile (88 bpm) — single extreme bradycardia outlier (71 bpm) while preserving normal range.
   - `oldpeak`  : Yeo-Johnson / log1p transformation + 99th percentile cap (4.2) — ST depression has strong positive skew (skew = 1.27) with extreme values up to 6.2.
   - `ca` & `thal` : Remove / re-impute invalid UCI anomaly codes (`ca == 4`, `thal == 0`).
"""
import numpy as np
import pandas as pd
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:
        def fit_transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    class DBSCAN:
        def __init__(self, eps=2.2, min_samples=5):
            self.eps = eps
            self.min_samples = min_samples
        def fit_predict(self, X):
            X = np.asarray(X, dtype=float)
            dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
            neighbors = (dists <= self.eps).sum(axis=1)
            core = neighbors >= self.min_samples
            labels = np.full(len(X), -1, dtype=int)
            for i in range(len(X)):
                if core[i] or (dists[i, core] <= self.eps).any():
                    labels[i] = 0
            return labels

CONT_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def detect_outliers_all_methods(df):
    """Run IQR, Z-Score, and multivariate DBSCAN outlier detection across continuous clinical features."""
    clean_df = df.drop_duplicates().reset_index(drop=True)
    X_cont = clean_df[CONT_COLS].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X_cont)

    dbscan = DBSCAN(eps=2.2, min_samples=5)
    db_labels = dbscan.fit_predict(X_scaled)
    db_outlier_mask = (db_labels == -1)

    rows = []
    per_col_masks = {}

    for i, col in enumerate(CONT_COLS):
        s = clean_df[col].astype(float)
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lo_iqr, hi_iqr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = (s < lo_iqr) | (s > hi_iqr)

        z_scores = np.abs((s - s.mean()) / (s.std() + 1e-9))
        z_mask = z_scores > 3.0

        # Multivariate DBSCAN outliers that also deviate (>2 std) on this specific feature
        col_db_mask = db_outlier_mask & (np.abs(X_scaled[:, i]) > 2.0)
        consensus_mask = (iqr_mask.astype(int) + z_mask.astype(int) + col_db_mask.astype(int)) >= 2

        per_col_masks[col] = {
            "iqr": iqr_mask,
            "z": z_mask,
            "dbscan": col_db_mask,
            "consensus": consensus_mask,
            "lo_iqr": lo_iqr,
            "hi_iqr": hi_iqr
        }

        rows.append({
            "Column": col,
            "Min": round(float(s.min()), 2),
            "Max": round(float(s.max()), 2),
            "Skewness": round(float(s.skew()), 3),
            "IQR Bounds": f"[{lo_iqr:.1f}, {hi_iqr:.1f}]",
            "IQR Outliers": int(iqr_mask.sum()),
            "Z-Score (|Z|>3)": int(z_mask.sum()),
            "DBSCAN Outliers": int(col_db_mask.sum()),
            "Consensus (>=2 Methods)": int(consensus_mask.sum())
        })

    summary_df = pd.DataFrame(rows)
    return summary_df, per_col_masks, int(db_outlier_mask.sum())


def apply_domain_outlier_strategy(df):
    """Apply case-by-case clinical domain outlier treatment and return (treated_df, decisions_df)."""
    work = df.drop_duplicates().reset_index(drop=True).copy()
    _, masks, _ = detect_outliers_all_methods(work)

    decisions = []

    # 1. age: Retain
    s_age = work["age"].astype(float)
    decisions.append({
        "Column": "age",
        "Detected Outliers (IQR / Z / DBSCAN)": f"{int(masks['age']['iqr'].sum())} / {int(masks['age']['z'].sum())} / {int(masks['age']['dbscan'].sum())}",
        "Selected Action": "Retain (No Capping)",
        "Domain Justification": "Adult cardiology ages (29-77 yrs) are physiologically valid with symmetric distribution (skew = -0.20).",
        "Skew Before": round(float(s_age.skew()), 3),
        "Skew After": round(float(s_age.skew()), 3)
    })

    # 2. trestbps: Cap (Winsorize) at [1st, 99th] percentile (94 to 180 mmHg)
    s_bp = work["trestbps"].astype(float)
    bp_lo, bp_hi = float(np.percentile(s_bp, 1)), float(np.percentile(s_bp, 99))
    work["trestbps"] = s_bp.clip(bp_lo, bp_hi).round().astype(int)
    decisions.append({
        "Column": "trestbps",
        "Detected Outliers (IQR / Z / DBSCAN)": f"{int(masks['trestbps']['iqr'].sum())} / {int(masks['trestbps']['z'].sum())} / {int(masks['trestbps']['dbscan'].sum())}",
        "Selected Action": f"Cap (Winsorize at [{bp_lo:.0f}, {bp_hi:.0f}] mmHg)",
        "Domain Justification": "Hypertensive crisis spikes (>180 mmHg, up to 200) are real patients and should not be dropped, but capping prevents leverage distortion.",
        "Skew Before": round(float(s_bp.skew()), 3),
        "Skew After": round(float(work["trestbps"].skew()), 3)
    })

    # 3. chol: Cap extreme tail at 99th percentile + add log1p transformed feature (`chol_log`)
    s_chol = work["chol"].astype(float)
    chol_hi = float(np.percentile(s_chol, 99))
    work["chol"] = s_chol.clip(lower=float(s_chol.min()), upper=chol_hi).round().astype(int)
    work["chol_log"] = np.log1p(s_chol).round(4)
    decisions.append({
        "Column": "chol",
        "Detected Outliers (IQR / Z / DBSCAN)": f"{int(masks['chol']['iqr'].sum())} / {int(masks['chol']['z'].sum())} / {int(masks['chol']['dbscan'].sum())}",
        "Selected Action": f"Cap (> {chol_hi:.0f} mg/dL) + Log Transform (chol_log)",
        "Domain Justification": "Severe hypercholesterolemia (e.g., 564 mg/dL) creates heavy right skew (1.14); capping + log1p stabilizes variance.",
        "Skew Before": round(float(s_chol.skew()), 3),
        "Skew After": round(float(work["chol_log"].skew()), 3)
    })

    # 4. thalach: Cap lower tail at 1st percentile (88 bpm)
    s_hr = work["thalach"].astype(float)
    hr_lo = float(np.percentile(s_hr, 1))
    work["thalach"] = s_hr.clip(lower=hr_lo, upper=float(s_hr.max())).round().astype(int)
    decisions.append({
        "Column": "thalach",
        "Detected Outliers (IQR / Z / DBSCAN)": f"{int(masks['thalach']['iqr'].sum())} / {int(masks['thalach']['z'].sum())} / {int(masks['thalach']['dbscan'].sum())}",
        "Selected Action": f"Cap Lower Tail (< {hr_lo:.0f} bpm)",
        "Domain Justification": "Single extreme low heart rate (71 bpm) is winsorized to 1st percentile while keeping high exercise heart rates intact.",
        "Skew Before": round(float(s_hr.skew()), 3),
        "Skew After": round(float(work["thalach"].skew()), 3)
    })

    # 5. oldpeak: Cap at 99th percentile (4.2) + add log1p transform (`oldpeak_log`)
    s_op = work["oldpeak"].astype(float)
    op_hi = float(np.percentile(s_op, 99))
    work["oldpeak"] = s_op.clip(0.0, op_hi).round(2)
    work["oldpeak_log"] = np.log1p(s_op).round(4)
    decisions.append({
        "Column": "oldpeak",
        "Detected Outliers (IQR / Z / DBSCAN)": f"{int(masks['oldpeak']['iqr'].sum())} / {int(masks['oldpeak']['z'].sum())} / {int(masks['oldpeak']['dbscan'].sum())}",
        "Selected Action": f"Cap (> {op_hi:.1f}) + Log Transform (oldpeak_log)",
        "Domain Justification": "ST depression is zero-inflated and strongly right-skewed (1.27, max 6.2); log1p + 99th percentile cap normalizes the tail.",
        "Skew Before": round(float(s_op.skew()), 3),
        "Skew After": round(float(work["oldpeak_log"].skew()), 3)
    })

    # 6. ca & thal invalid UCI codes (ca=4, thal=0): Remove / Replace with domain mode
    bad_codes = int(((work["ca"] == 4) | (work["thal"] == 0)).sum())
    work.loc[work["ca"] == 4, "ca"] = int(work.loc[work["ca"] != 4, "ca"].mode()[0])
    work.loc[work["thal"] == 0, "thal"] = int(work.loc[work["thal"] != 0, "thal"].mode()[0])
    decisions.append({
        "Column": "ca & thal (Anomaly Codes)",
        "Detected Outliers (IQR / Z / DBSCAN)": f"{bad_codes} invalid code rows (ca=4, thal=0)",
        "Selected Action": "Remove Invalid Codes & Replace with Valid Mode",
        "Domain Justification": "In UCI Cleveland documentation, ca=4 and thal=0 are unrecorded '?' codes rather than physiological categories.",
        "Skew Before": None,
        "Skew After": None
    })

    return work, pd.DataFrame(decisions)
