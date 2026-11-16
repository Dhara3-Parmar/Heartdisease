"""
HeartCare - Missing Value Imputation & Statistical Method Selection (KNN, MICE, Predictive Modeling).

1. Hidden Missingness Detection in heart.csv :
   - Standard UCI Heart Disease documentation specifies `ca` in 0..3 and `thal` in 1..3.
   - In `heart.csv`, `ca == 4` (5 rows) and `thal == 0` (2 rows) are undocumented anomaly codes representing original UCI '?' missing values.
2. Controlled Missingness Benchmark (15% MCAR simulation) :
   - Evaluates 7 clinical columns (continuous: age, trestbps, chol, thalach, oldpeak; categorical: ca, thal)
   - Compares KNN Imputation, MICE (IterativeImputer), and Predictive Modeling (RandomForestRegressor / Classifier)
3. Statistical Analysis & Manual Per-Column Selection :
   - Continuous columns: RMSE, MAE, Mean Diff, Std Diff, Kolmogorov-Smirnov (KS) test p-value
   - Categorical columns: Imputation Accuracy (%), Macro-F1, KS statistic
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score

from confidence import FEATURES

CONT_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["ca", "thal"]
EVAL_COLS = CONT_COLS + CAT_COLS


def detect_hidden_missing(df):
    """Audit explicit NaNs and domain-encoded missing codes (ca=4, thal=0) in heart.csv."""
    rows = []
    for f in FEATURES:
        explicit_na = int(df[f].isna().sum())
        hidden_na = 0
        note = "No missing values"
        if f == "ca":
            hidden_na = int((df[f] == 4).sum())
            note = "Code 4 represents '?' (missing) in original UCI Cleveland data"
        elif f == "thal":
            hidden_na = int((df[f] == 0).sum())
            note = "Code 0 represents '?' (missing) in original UCI Cleveland data"
        rows.append({
            "Feature": f,
            "Explicit NaN": explicit_na,
            "Hidden Anomaly Missing": hidden_na,
            "Total Missing": explicit_na + hidden_na,
            "Pct Missing (%)": round((explicit_na + hidden_na) / len(df) * 100, 2),
            "Domain Note": note
        })
    return pd.DataFrame(rows)


def prepare_clean_reference(df):
    """Convert ca=4 and thal=0 to NaN and return (df_with_nan, complete_cases)."""
    df_clean = df.drop_duplicates().copy()
    df_with_nan = df_clean.copy()
    df_with_nan.loc[df_with_nan["ca"] == 4, "ca"] = np.nan
    df_with_nan.loc[df_with_nan["thal"] == 0, "thal"] = np.nan
    complete_cases = df_with_nan.dropna().reset_index(drop=True)
    return df_with_nan, complete_cases


def _postprocess_categories(df_imp, ref_df):
    """Round/clip imputed categorical columns to valid allowed category codes."""
    out = df_imp.copy()
    for f in FEATURES:
        if f not in CONT_COLS:
            allowed = np.sort(ref_df[f].dropna().unique())
            vals = out[f].to_numpy(dtype=float)
            idx = np.abs(vals[:, None] - allowed[None, :]).argmin(axis=1)
            out[f] = allowed[idx]
    return out


def run_knn_imputation(df_missing, ref_df, n_neighbors=5):
    mins = ref_df[FEATURES].min()
    span = (ref_df[FEATURES].max() - mins).replace(0, 1.0)
    scaled = (df_missing[FEATURES] - mins) / span
    imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    res_scaled = pd.DataFrame(imp.fit_transform(scaled), columns=FEATURES, index=df_missing.index)
    res = res_scaled * span + mins
    return _postprocess_categories(res, ref_df)


def run_mice_imputation(df_missing, ref_df, max_iter=15, seed=42):
    imp = IterativeImputer(max_iter=max_iter, random_state=seed)
    res = pd.DataFrame(imp.fit_transform(df_missing[FEATURES]), columns=FEATURES, index=df_missing.index)
    return _postprocess_categories(res, ref_df)


def run_predictive_imputation(df_missing, ref_df, seed=42):
    out = df_missing[FEATURES].copy()
    col_medians = ref_df[FEATURES].median()
    for f in FEATURES:
        miss_mask = out[f].isna()
        if not miss_mask.any():
            continue
        other = [c for c in FEATURES if c != f]
        obs_mask = ~miss_mask
        X_train = out.loc[obs_mask, other].fillna(col_medians[other])
        y_train = out.loc[obs_mask, f]
        X_pred = out.loc[miss_mask, other].fillna(col_medians[other])
        if f in CONT_COLS:
            model = RandomForestRegressor(n_estimators=60, max_depth=6, random_state=seed)
            model.fit(X_train, y_train)
            out.loc[miss_mask, f] = model.predict(X_pred)
        else:
            model = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=seed)
            model.fit(X_train.astype(float), y_train.astype(int))
            out.loc[miss_mask, f] = model.predict(X_pred.astype(float))
    return _postprocess_categories(out, ref_df)


def evaluate_imputation_methods(df, missing_rate=0.15, seed=42):
    """Benchmark KNN, MICE, and Predictive RF under 15% simulated missingness and impute heart.csv."""
    _, complete = prepare_clean_reference(df)
    rng = np.random.default_rng(seed)
    sim_missing = complete[FEATURES].copy().astype(float)
    masks = {}
    for col in EVAL_COLS:
        mask = rng.random(len(sim_missing)) < missing_rate
        if mask.sum() < 5:
            mask[:5] = True
        masks[col] = mask
        sim_missing.loc[mask, col] = np.nan

    imputed_sets = {
        "KNN Imputation": run_knn_imputation(sim_missing, complete),
        "MICE (IterativeImputer)": run_mice_imputation(sim_missing, complete, seed=seed),
        "Predictive Modeling (RF)": run_predictive_imputation(sim_missing, complete, seed=seed),
    }

    records = []
    for col in EVAL_COLS:
        mask = masks[col]
        y_true = complete.loc[mask, col].to_numpy(dtype=float)
        full_true = complete[col].to_numpy(dtype=float)
        is_cont = col in CONT_COLS

        for method, df_imp in imputed_sets.items():
            y_pred = df_imp.loc[mask, col].to_numpy(dtype=float)
            full_imp = df_imp[col].to_numpy(dtype=float)
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mean_bias = float(abs(full_imp.mean() - full_true.mean()))
            std_bias = float(abs(full_imp.std() - full_true.std()))
            ks_stat, ks_pval = ks_2samp(full_true, full_imp)

            if is_cont:
                nrmse = rmse / (full_true.std() + 1e-9)
                composite = 0.65 * nrmse + 0.35 * float(ks_stat)
                acc = np.nan
                f1 = np.nan
            else:
                acc = float(accuracy_score(y_true.astype(int), y_pred.astype(int)))
                f1 = float(f1_score(y_true.astype(int), y_pred.astype(int), average="macro", zero_division=0))
                composite = (1.0 - acc) + 0.35 * float(ks_stat)

            records.append({
                "Column": col,
                "Type": "Continuous" if is_cont else "Categorical",
                "Method": method,
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "Accuracy (%)": round(acc * 100, 1) if not np.isnan(acc) else None,
                "Macro-F1": round(f1, 3) if not np.isnan(f1) else None,
                "Mean Diff": round(mean_bias, 4),
                "Std Diff": round(std_bias, 4),
                "KS Stat": round(float(ks_stat), 4),
                "KS p-value": round(float(ks_pval), 4),
                "Selection Score (Lower=Better)": round(composite, 4),
            })

    eval_df = pd.DataFrame(records)
    best_idx = eval_df.groupby("Column")["Selection Score (Lower=Better)"].idxmin()
    best_df = eval_df.loc[best_idx].sort_values(["Type", "Column"]).reset_index(drop=True)

    rationale = []
    for _, row in best_df.iterrows():
        c, m, t = row["Column"], row["Method"], row["Type"]
        if t == "Continuous":
            reason = f"Lowest composite error (RMSE={row['RMSE']}, MAE={row['MAE']}) and preserves distribution (KS p={row['KS p-value']})"
        else:
            reason = f"Highest categorical recovery (Acc={row['Accuracy (%)']}%, Macro-F1={row['Macro-F1']}) with minimal distribution shift"
        rationale.append({
            "Column": c,
            "Type": t,
            "Selected Best Method": m,
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "Accuracy (%)": row["Accuracy (%)"],
            "KS p-value": row["KS p-value"],
            "Statistical Justification": reason
        })
    selection_df = pd.DataFrame(rationale)

    df_real_missing, complete_ref = prepare_clean_reference(df)
    real_imputed_by_method = {
        "KNN Imputation": run_knn_imputation(df_real_missing, complete_ref),
        "MICE (IterativeImputer)": run_mice_imputation(df_real_missing, complete_ref, seed=seed),
        "Predictive Modeling (RF)": run_predictive_imputation(df_real_missing, complete_ref, seed=seed),
    }
    final_heart_df = df_real_missing.copy()
    for _, row in selection_df.iterrows():
        col = row["Column"]
        chosen = row["Selected Best Method"]
        final_heart_df[col] = real_imputed_by_method[chosen][col]

    return eval_df, selection_df, final_heart_df
