import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from src.models.model_utils import ModelUtils, get_class_weights_info

import joblib
import json


# --------------------------------------------------------------
# Helper: Sweep thresholds to find most balanced severe split
# --------------------------------------------------------------
def sweep_thresholds(county_df, pct_col, thresholds):
    results = []
    total = len(county_df)

    for t in thresholds:
        severe = (county_df[pct_col] >= t).sum()
        non_severe = total - severe
        ratio = severe / total if total > 0 else 0

        results.append({
            "threshold": t,
            "non_severe": non_severe,
            "severe": severe,
            "severe_ratio": ratio,
            "balance_score": abs(0.5 - ratio)  # closer to 0 => more balanced
        })

    return pd.DataFrame(results)


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    print("=== TX COUNTY SEVERE OUTAGE CLASSIFICATION (HARRIS & BEXAR) ===")

    utils = ModelUtils()
    df, selected_features = utils.load_data_with_features()

    # --------------------------
    # Counties to process
    # --------------------------
    target_counties = [
        {"fips": 48201, "name": "Harris_County_TX"},
        {"fips": 48029, "name": "Bexar_County_TX"},
    ]

    results_dir = utils.results_dir / "county_models_tx"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Candidate thresholds for Texas
    threshold_candidates = [
        0.0001,
        0.00015,
        0.0002,
        0.00025,
        0.0003,
        0.0004,
        0.0005,
    ]

    all_results = {}
    pct_col = "pct_out_area_unified"

    # Hyperparameter search space
    param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000, 2000],
        "class_weight": ["balanced"],
    }

    # ----------------------------------------------------------
    # Loop over counties
    # ----------------------------------------------------------
    for county in target_counties:
        fips = county["fips"]
        name = county["name"]

        print("\n" + "=" * 60)
        print(f"Processing {name} ({fips})")
        print("=" * 60)

        county_df = df[df["fips_code"] == fips].copy()
        if len(county_df) == 0:
            print("No data for county. Skipping.")
            continue

        # ----------------------------------------------------------
        # THRESHOLD SWEEP
        # ----------------------------------------------------------
        print("\n--- THRESHOLD SWEEP ---")
        threshold_results = sweep_thresholds(county_df, pct_col, threshold_candidates)

        print(f"\n{'Threshold':>12} | {'Non-Severe':>12} | {'Severe':>12} | {'Severe %':>10}")
        print("-" * 60)
        for _, row in threshold_results.iterrows():
            print(
                f"{row['threshold']:>12.5f} | "
                f"{row['non_severe']:>12,} | "
                f"{row['severe']:>12,} | "
                f"{row['severe_ratio']*100:>9.2f}%"
            )

        # Pick the threshold with the most balanced severe ratio
        best_idx = threshold_results["balance_score"].idxmin()
        best_threshold = threshold_results.loc[best_idx, "threshold"]
        best_ratio = threshold_results.loc[best_idx, "severe_ratio"]

        print(f"\n>>> Best Balanced Threshold for {name}: {best_threshold:.5f}")
        print(f"    Severe Ratio: {best_ratio:.2%}")

        # ----------------------------------------------------------
        # CREATE FINAL TARGET USING BEST THRESHOLD
        # ----------------------------------------------------------
        county_df["severe_outage"] = (county_df[pct_col] >= best_threshold).astype(int)
        target_col = "severe_outage"

        # ----------------------------------------------------------
        # YEAR SPLIT (Train/Val = ≠2022, Test = 2022)
        # ----------------------------------------------------------
        test_df = county_df[county_df["year"] == 2022].copy()
        train_val_df = county_df[county_df["year"] != 2022].copy()

        if train_val_df[target_col].nunique() < 2:
            print("Training data has only one class. Skipping county.")
            continue

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.20,
            random_state=42,
            stratify=train_val_df[target_col],
        )

        print(f"\nTrain size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

        X_train = train_df[selected_features]
        y_train = train_df[target_col]

        X_val = val_df[selected_features]
        y_val = val_df[target_col]

        X_test = test_df[selected_features]
        y_test = test_df[target_col]

        # ----------------------------------------------------------
        # HYPERPARAMETER TUNING
        # ----------------------------------------------------------
        print("\n--- HYPERPARAMETER TUNING ---")

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=LogisticRegression(random_state=42),
            param_distributions=param_distributions,
            n_iter=20,
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print("Best Parameters:", search.best_params_)

        # ----------------------------------------------------------
        # EVALUATION ON 2022
        # ----------------------------------------------------------
        print("\n--- EVALUATION (2022 Test Set) ---")

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        if len(y_test.unique()) > 1:
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        else:
            roc_auc = pr_auc = 0.0
            print("Warning: Only one class in test set.")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC:  {roc_auc:.4f}")
        print(f"PR  AUC:  {pr_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # ----------------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------------
        result = {
            "best_threshold": best_threshold,
            "severe_ratio": best_ratio,
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            "test_metrics": {
                "accuracy": acc,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "f1": f1,
            },
            "report": classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            ),
            "confusion_matrix": cm.tolist(),
            "threshold_results": threshold_results.to_dict(orient="records"),
        }

        all_results[name] = result

        # Save model
        model_dir = utils.models_dir / "county_models_tx"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_dir / f"{name}_severe_logistic_regression.pkl")

        # Save predictions
        preds_df = test_df[["fips_code", "year", "month", "day_of_year", pct_col]].copy()
        preds_df["actual"] = y_test
        preds_df["predicted"] = y_pred
        preds_df["probability"] = y_proba
        preds_df["threshold_used"] = best_threshold
        preds_df.to_csv(results_dir / f"{name}_2022_predictions.csv", index=False)

    # Save Summary File
    with open(results_dir / "tx_severe_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nProcessing Complete! TX results saved.")


if __name__ == "__main__":
    main()
