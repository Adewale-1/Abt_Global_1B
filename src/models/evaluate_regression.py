"""
Regression Evaluation Script (pct_out_area_unified)
===================================================

Outputs:
- results/models/linear_regression_evaluation.txt
- results/models/linear_regression_predictions.csv  (incl. residuals)
- results/figures/linear_regression/error_by_county_rmse.png
- results/figures/linear_regression/error_by_county_mae.png
- results/figures/linear_regression/error_by_month_rmse.png
- results/figures/linear_regression/error_by_month_mae.png
- results/figures/linear_regression/error_by_target_decile.png
- results/figures/linear_regression/residuals_qq.png
- results/figures/linear_regression/residuals_acf.png
- results/models/linear_regression_diagnostics.txt  (DW, BP if available)
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.model_utils import ModelUtils

# Optional statsmodels/scipy diagnostics
try:
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

sns.set_style("whitegrid")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("=" * 80)
    print("REGRESSION MODEL EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    utils = ModelUtils()

    # Load trained model + params
    print("STEP 1: LOAD MODEL")
    print("-" * 80)
    model, params = utils.load_model("linear_regression_best")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Load data & regression features
    print("\nSTEP 2: LOAD DATA")
    print("-" * 80)
    df = pd.read_csv(utils.data_dir / "merged_weather_outages_2019_2024_encoded.csv", low_memory=False)

    feat_path = utils.results_dir / "feature_selection" / "selected_features_regression.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing regression features at {feat_path}")
    selected_features = pd.read_csv(feat_path)["feature"].tolist()

    # Use temporal split; we only need the test set for final evaluation
    train_df, val_df, test_df = utils.temporal_split(df, target_col="any_out")

    target_col = "pct_out_area_unified"
    X_test = test_df[selected_features].copy()
    y_test = test_df[target_col].copy()

    if X_test.isnull().sum().sum() > 0:
        X_test.fillna(0, inplace=True)

    # Predictions
    print("\nSTEP 3: PREDICT")
    print("-" * 80)
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    # Metrics
    print("\nSTEP 4: METRICS")
    print("-" * 80)
    _rmse = rmse(y_test, y_pred)
    _mae = float(mean_absolute_error(y_test, y_pred))
    _r2 = float(r2_score(y_test, y_pred))
    print(f"Test RMSE: {_rmse:.4f} | MAE: {_mae:.4f} | R²: {_r2:.4f}")

    # Save predictions CSV (with metadata if present)
    print("\nSTEP 5: SAVE PREDICTIONS")
    print("-" * 80)
    meta_cols = [c for c in ["fips_code", "year", "month", "day_of_year", "season"] if c in test_df.columns]
    preds_df = test_df[meta_cols].copy()
    preds_df["y_true"] = y_test.values
    preds_df["y_pred"] = y_pred
    preds_df["residual"] = preds_df["y_true"] - preds_df["y_pred"]
    pred_path = utils.results_dir / "models" / "linear_regression_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(pred_path, index=False)
    print(f"Saved predictions: {pred_path}")

    # ============================
    # Error by county & season/month heatmaps
    # ============================
    print("\nSTEP 6: ERROR HEATMAPS (COUNTY / MONTH)")
    print("-" * 80)
    fig_dir = utils.results_dir / "figures" / "linear_regression"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Helper to plot heatmap from pivot
    def _heatmap(pivot_df, title, save_name):
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="magma", cbar=True)
        plt.title(title)
        plt.tight_layout()
        path = fig_dir / save_name
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # County × Season (if available)
    if "fips_code" in preds_df.columns and "season" in preds_df.columns:
        g1 = preds_df.groupby(["fips_code", "season"]).apply(
            lambda g: pd.Series({
                "RMSE": rmse(g["y_true"], g["y_pred"]),
                "MAE": float(mean_absolute_error(g["y_true"], g["y_pred"]))
            }),
            include_groups=False  # <-- future-proof pandas behavior
        ).reset_index()
        for metric in ["RMSE", "MAE"]:
            pv = g1.pivot(index="fips_code", columns="season", values=metric).fillna(np.nan)
            _heatmap(pv, f"{metric} by County × Season (Test 2024)", f"error_by_county_{metric.lower()}.png")

    # County × Month
    if "fips_code" in preds_df.columns and "month" in preds_df.columns:
        g2 = preds_df.groupby(["fips_code", "month"]).apply(
            lambda g: pd.Series({
                "RMSE": rmse(g["y_true"], g["y_pred"]),
                "MAE": float(mean_absolute_error(g["y_true"], g["y_pred"]))
            }),
            include_groups=False  # <-- future-proof pandas behavior
        ).reset_index()
        for metric in ["RMSE", "MAE"]:
            pv = g2.pivot(index="fips_code", columns="month", values=metric).fillna(np.nan)
            _heatmap(pv, f"{metric} by County × Month (Test 2024)", f"error_by_month_{metric.lower()}.png")

    # ============================
    # Target-magnitude calibration (decile error)
    # ============================
    print("\nSTEP 7: TARGET-MAGNITUDE CALIBRATION (DECILES)")
    print("-" * 80)
    if preds_df["y_true"].nunique() > 10:
        preds_df["true_decile"] = pd.qcut(preds_df["y_true"], 10, labels=False, duplicates="drop")
    else:
        preds_df["true_decile"] = pd.cut(preds_df["y_true"], bins=10, labels=False, include_lowest=True)

    preds_df["abs_perc_error"] = np.where(
        preds_df["y_true"] != 0,
        np.abs(preds_df["residual"] / preds_df["y_true"]),
        np.nan
    )

    dec = preds_df.groupby("true_decile").agg(
        MAE=("residual", lambda x: float(np.mean(np.abs(x)))),
        MAPE=("abs_perc_error", "mean")
    ).reset_index()

    plt.figure(figsize=(10, 5))
    plt.bar(dec["true_decile"], dec["MAE"], label="MAE")
    ax2 = plt.twinx()
    ax2.plot(dec["true_decile"], dec["MAPE"], marker="o", label="MAPE", linestyle="--")
    plt.title("Error by Target Decile (MAE bars, MAPE line)")
    plt.xlabel("Decile of True pct_out_area_unified")
    plt.tight_layout()
    path = fig_dir / "error_by_target_decile.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ============================
    # Residual diagnostics: QQ, ACF, DW, BP
    # ============================
    print("\nSTEP 8: RESIDUAL DIAGNOSTICS")
    print("-" * 80)

    # QQ plot (if scipy available)
    if HAS_SCIPY:
        plt.figure(figsize=(6, 6))
        scipy_stats.probplot(preds_df["residual"], dist="norm", plot=plt)
        plt.title("Residuals QQ-Plot")
        path = fig_dir / "residuals_qq.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
    else:
        print("SciPy not available; skipping QQ plot.")

    # Simple ACF (0..40 lags) using numpy
    def _acf(x, nlags=40):
        x = np.asarray(x) - np.mean(x)
        autocorr = np.correlate(x, x, mode="full")[len(x)-1:len(x)-1+nlags+1]
        return autocorr / autocorr[0] if autocorr[0] != 0 else np.zeros(nlags+1)

    acf_vals = _acf(preds_df["residual"].values, nlags=40)
    plt.figure(figsize=(10, 4))
    markerline, stemlines, baseline = plt.stem(range(len(acf_vals)), acf_vals)  # removed use_line_collection
    try:
        baseline.set_visible(False)  # cosmetic
    except Exception:
        pass
    plt.title("Residuals Autocorrelation (ACF)")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    path = fig_dir / "residuals_acf.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # DW and BP stats (if statsmodels available)
    diag_lines = []
    if HAS_SM:
        dw = durbin_watson(preds_df["residual"])
        diag_lines.append(f"Durbin-Watson: {dw:.4f}")

        # Breusch-Pagan: need exog with constant
        try:
            X_exog = sm.add_constant(X_test[selected_features], has_constant="add")
            bp_lm, bp_lmpval, bp_f, bp_fpval = het_breuschpagan(preds_df["residual"], X_exog)
            diag_lines.append(f"Breusch-Pagan LM: {bp_lm:.4f}, p-value: {bp_lmpval:.6f}")
            diag_lines.append(f"Breusch-Pagan F : {bp_f:.4f}, p-value: {bp_fpval:.6f}")
        except Exception as e:
            diag_lines.append(f"Breusch-Pagan test failed: {e}")
    else:
        diag_lines.append("statsmodels not available; skipping Durbin–Watson and Breusch–Pagan.")

    diag_path = utils.results_dir / "models" / "linear_regression_diagnostics.txt"
    with open(diag_path, "w") as f:
        f.write("\n".join(diag_lines))
    print(f"Saved diagnostics: {diag_path}")

    # ============================
    # Text evaluation report
    # ============================
    print("\nSTEP 9: SAVE EVALUATION REPORT")
    print("-" * 80)
    report = [
        "=" * 80,
        "LINEAR REGRESSION EVALUATION REPORT",
        "=" * 80,
        f"Evaluated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Test Period: 2024",
        "",
        "PRIMARY METRICS",
        "-" * 80,
        f"RMSE: {_rmse:.4f}",
        f"MAE : {_mae:.4f}",
        f"R²  : {_r2:.4f}",
        "",
        "ARTIFACTS",
        "-" * 80,
        f"Predictions CSV: {pred_path}",
        "County×Season heatmaps: error_by_county_rmse.png / error_by_county_mae.png",
        "County×Month  heatmaps: error_by_month_rmse.png / error_by_month_mae.png",
        f"Decile error: {fig_dir / 'error_by_target_decile.png'}",
        f"Residual QQ: {fig_dir / 'residuals_qq.png'}",
        f"Residual ACF: {fig_dir / 'residuals_acf.png'}",
        f"Diagnostics: {diag_path}",
        "",
        "NOTES",
        "-" * 80,
        "Use error-by-county/month heatmaps to identify hotspots of under/over-prediction.",
        "Decile plot checks calibration across outage magnitude—watch for high-MAPE in top deciles.",
        "DW≈2 suggests no autocorrelation; BP p<0.05 indicates heteroscedasticity.",
    ]
    utils.save_results("\n".join(report), filename="linear_regression_evaluation.txt")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"Key Results -> RMSE: {_rmse:.4f} | MAE: {_mae:.4f} | R²: {_r2:.4f}")
    print("Next: python src/models/visualize_regression_results.py")


if __name__ == "__main__":
    main()
