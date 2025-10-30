"""
Regression Visualization Script
===============================

Produces:
- Parity plot (y_true_vs_y_pred.png)
- Residuals histogram (residuals_hist.png)
- Residuals vs fitted (residuals_vs_fitted.png)
- Top coefficients bar chart (top_coefficients.png)  [if model has coef_]
- Permutation importance bar chart + CSV:
    - results/figures/linear_regression/permutation_importance.png
    - results/models/permutation_importance.csv
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_utils import ModelUtils

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def main():
    print("=" * 80)
    print("REGRESSION VISUALIZATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    utils = ModelUtils()

    # Paths
    fig_dir = utils.results_dir / "figures" / "linear_regression"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load model & data
    print("\nLOADING MODEL AND DATA")
    print("-" * 80)
    model, _ = utils.load_model("linear_regression_best")

    df = pd.read_csv(utils.data_dir / "merged_weather_outages_2019_2024_encoded.csv", low_memory=False)
    feat_path = utils.results_dir / "feature_selection" / "selected_features_regression.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing regression features at {feat_path}")
    selected_features = pd.read_csv(feat_path)["feature"].tolist()

    train_df, val_df, test_df = utils.temporal_split(df, target_col="any_out")
    target_col = "pct_out_area_unified"

    X_test = test_df[selected_features].copy()
    y_test = test_df[target_col].copy()
    if X_test.isnull().sum().sum() > 0:
        X_test.fillna(0, inplace=True)

    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    print(f"\nGenerating visualizations for {len(X_test):,} test samples...")

    # -----------------------------
    # 1) Parity plot
    # -----------------------------
    print("\n1) Parity plot...")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, edgecolor="none")
    lims = [
        min(np.min(y_test), np.min(y_pred)),
        max(np.max(y_test), np.max(y_pred)),
    ]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True pct_out_area_unified")
    plt.ylabel("Predicted")
    plt.title("Parity Plot (y vs ŷ)")
    plt.tight_layout()
    path = fig_dir / "y_true_vs_y_pred.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # -----------------------------
    # 2) Residuals histogram
    # -----------------------------
    print("\n2) Residuals histogram...")
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=60, alpha=0.8)
    plt.xlabel("Residual (y - ŷ)")
    plt.ylabel("Count")
    plt.title("Residuals Histogram")
    plt.tight_layout()
    path = fig_dir / "residuals_hist.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # -----------------------------
    # 3) Residuals vs fitted
    # -----------------------------
    print("\n3) Residuals vs fitted...")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.3, edgecolor="none")
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Fitted (ŷ)")
    plt.ylabel("Residual (y - ŷ)")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    path = fig_dir / "residuals_vs_fitted.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # -----------------------------
    # 4) Top coefficients (if available)
    # -----------------------------
    if hasattr(model, "named_steps"):
        final_est = model.named_steps.get("model", model)
    else:
        final_est = model

    if hasattr(final_est, "coef_"):
        print("\n4) Top coefficients...")
        coefs = pd.DataFrame({"feature": selected_features, "coef": final_est.coef_.ravel()})
        coefs["abs_coef"] = coefs["coef"].abs()
        top = coefs.sort_values("abs_coef", ascending=False).head(20)

        plt.figure(figsize=(10, 7))
        colors = ["green" if c > 0 else "red" for c in top["coef"]]
        plt.barh(range(len(top)), top["coef"], alpha=0.8, color=colors)
        plt.yticks(range(len(top)), top["feature"])
        plt.gca().invert_yaxis()
        plt.xlabel("Coefficient")
        plt.title("Top 20 Coefficients (ElasticNet)")
        plt.tight_layout()
        path = fig_dir / "top_coefficients.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")
    else:
        print("\nModel does not expose coef_; skipping coefficient plot.")

    # -----------------------------
    # 5) Permutation importance (model-agnostic)
    # -----------------------------
    print("\n5) Permutation importance...")
    pi = permutation_importance(
        model, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=10, random_state=42, n_jobs=-1
    )
    pi_df = pd.DataFrame({
        "feature": selected_features,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values("importance_mean", ascending=False)

    # Save CSV
    pi_csv = utils.results_dir / "models" / "permutation_importance.csv"
    pi_df.to_csv(pi_csv, index=False)
    print(f"Saved permutation importance CSV: {pi_csv}")

    # Plot top 25
    topN = pi_df.head(25)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(topN)), topN["importance_mean"])
    plt.yticks(range(len(topN)), topN["feature"])
    plt.gca().invert_yaxis()
    plt.xlabel("Decrease in score (−RMSE)")
    plt.title("Permutation Importance (Top 25)")
    plt.tight_layout()
    path = fig_dir / "permutation_importance.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"Artifacts saved under: {fig_dir}")


if __name__ == "__main__":
    main()
