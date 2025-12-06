"""
ElasticNet-based Multiple Linear Regression Training with Time-Aware CV
======================================================================

- Uses regression feature list from results/feature_selection/selected_features_regression.csv
- Temporal holdout: Train(2019-2022) / Val(2023) / Test(2024) via ModelUtils.temporal_split
- Rolling-origin CV with TimeSeriesSplit(n_splits=5)
- Pipeline: StandardScaler -> ElasticNet
- Primary score: neg_root_mean_squared_error (RMSE)
- Saves:
  - models/trained/linear_regression_best.pkl (+ _params.json, _info.json)
  - results/models/linear_regression_training_log.txt
  - results/models/linear_regression_cv_summary.csv
  - results/models/linear_regression_cv_plot.png
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_utils import ModelUtils


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    print("=" * 80)
    print("LINEAR REGRESSION (ELASTICNET) MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    utils = ModelUtils()

    # -----------------------------
    # Load data & regression features
    # -----------------------------
    print("LOADING DATA")
    print("-" * 80)
    data_path = utils.data_dir / "merged_weather_outages_2019_2024_encoded.csv"
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Read regression-selected features
    feat_path = utils.results_dir / "feature_selection" / "selected_features_regression.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing regression features at {feat_path}. "
            f"Run src/preprocessing/enhanced_feature_selection_regression.py first."
        )
    selected_features = pd.read_csv(feat_path)["feature"].tolist()
    print(f"Using {len(selected_features)} regression features")

    # Temporal split (same as logistic workflow)
    print("\nTEMPORAL SPLIT")
    print("-" * 80)
    train_df, val_df, test_df = utils.temporal_split(df, target_col="any_out")  # split logic uses 'year' column

    # Prepare X/y with *regression* target
    target_col = "pct_out_area_unified"
    X_train = train_df[selected_features].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[selected_features].copy()
    y_val = val_df[target_col].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df[target_col].copy()

    # Handle any missing (should be minimal post-encoding)
    for X in (X_train, X_val, X_test):
        if X.isnull().sum().sum() > 0:
            X.fillna(0, inplace=True)

    # -----------------------------
    # CV setup (rolling-origin)
    # -----------------------------
    print("\nHYPERPARAMETER TUNING SETUP")
    print("-" * 80)
    tscv = TimeSeriesSplit(n_splits=5)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", ElasticNet(max_iter=5000, random_state=42)),
        ]
    )

    param_distributions = {
        "model__alpha": np.logspace(-4, 1, 20),      # regularization strength
        "model__l1_ratio": np.linspace(0.0, 1.0, 11) # 0=ridge,1=lasso
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=30,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True,
    )

    print("Searching 30 random ElasticNet configs with TimeSeriesSplit(5)")
    # -----------------------------
    # Fit search
    # -----------------------------
    print("\nTRAINING WITH ROLLING-ORIGIN CV")
    print("-" * 80)
    search.fit(X_train, y_train)

    print("\nTraining completed!")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")
    print("Best Hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  {k} = {v}")

    best_model = search.best_estimator_

    # -----------------------------
    # Manual per-fold metric table (on training period)
    # -----------------------------
    print("\nCROSS-VALIDATION SUMMARY")
    print("-" * 80)
    cv_rows = []
    split_id = 0
    for train_idx, val_idx in tscv.split(X_train):
        split_id += 1
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        best_model.fit(X_tr, y_tr)
        preds = best_model.predict(X_va)
        row = {
            "fold": split_id,
            "rmse": rmse(y_va, preds),
            "mae": mean_absolute_error(y_va, preds),
            "r2": r2_score(y_va, preds),
            "n_train": len(X_tr),
            "n_val": len(X_va),
        }
        cv_rows.append(row)
        print(f"Fold {split_id}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.loc["mean"] = ["mean",
                         cv_df["rmse"].mean(),
                         cv_df["mae"].mean(),
                         cv_df["r2"].mean(),
                         cv_df["n_train"].mean(),
                         cv_df["n_val"].mean()]
    cv_df.loc["std"] = ["std",
                        cv_df["rmse"].std(),
                        cv_df["mae"].std(),
                        cv_df["r2"].std(),
                        np.nan, np.nan]

    cv_path = utils.results_dir / "models" / "linear_regression_cv_summary.csv"
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(cv_path, index=True)
    print(f"Saved CV table: {cv_path}")

    # Plot RMSE by fold
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cv_df.iloc[:-2]["fold"], cv_df.iloc[:-2]["rmse"], marker="o")
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE (val)")
    ax.set_title("Rolling-Origin CV: RMSE by Fold")
    ax.grid(True, alpha=0.3)
    plot_path = utils.results_dir / "models" / "linear_regression_cv_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CV plot: {plot_path}")

    # -----------------------------
    # Validation & Test metrics (holdout years)
    # -----------------------------
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    val_rmse = rmse(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    test_rmse = rmse(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nVALIDATION SET PERFORMANCE (2023)")
    print("-" * 80)
    print(f"RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

    print("\nTEST SET PERFORMANCE (2024)")
    print("-" * 80)
    print(f"RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")

    # -----------------------------
    # Save model + metadata
    # -----------------------------
    training_info = {
        "model_type": "ElasticNet (Linear Regression family)",
        "pipeline_steps": ["StandardScaler", "ElasticNet"],
        "n_features": len(selected_features),
        "feature_names": selected_features,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "cv": "TimeSeriesSplit(n_splits=5)",
        "best_cv_neg_rmse": float(search.best_score_),
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
        "val_r2": float(val_r2),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
    }

    utils.save_model(
        model=best_model,
        model_name="linear_regression_best",
        hyperparameters=search.best_params_,
        training_info=training_info,
    )

    # Training log
    lines = [
        "=" * 80,
        "LINEAR REGRESSION TRAINING LOG",
        "=" * 80,
        f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "BEST HYPERPARAMETERS",
        "-" * 80,
    ]
    for k, v in search.best_params_.items():
        lines.append(f"{k}: {v}")
    lines.extend([
        "",
        "VALIDATION (2023)",
        "-" * 80,
        f"RMSE: {val_rmse:.4f}",
        f"MAE:  {val_mae:.4f}",
        f"R²:   {val_r2:.4f}",
        "",
        "TEST (2024)",
        "-" * 80,
        f"RMSE: {test_rmse:.4f}",
        f"MAE:  {test_mae:.4f}",
        f"R²:   {test_r2:.4f}",
        "",
        "CROSS-VALIDATION (TRAIN YEARS 2019–2022)",
        "-" * 80,
        cv_df.to_string(index=True),
        "",
    ])
    utils.save_results("\n".join(lines), filename="linear_regression_training_log.txt")

    print("\n")
    print("TRAINING COMPLETE!")
    print("Next steps:")
    print("  Evaluate: python src/models/evaluate_regression.py")
    print("  Visualize: python src/models/visualize_regression_results.py")


if __name__ == "__main__":
    main()
