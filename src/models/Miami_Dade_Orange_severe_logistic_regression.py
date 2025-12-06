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
from datetime import datetime


def find_balanced_threshold(pct_series, thresholds):
    """Find threshold that creates most balanced severe/non-severe split."""
    results = []
    for t in thresholds:
        severe = (pct_series >= t).sum()
        non_severe = (pct_series < t).sum()
        total = len(pct_series)
        ratio = severe / total if total > 0 else 0
        balance_score = abs(0.5 - ratio)
        results.append(
            {
                "threshold": t,
                "non_severe": non_severe,
                "severe": severe,
                "severe_ratio": ratio,
                "balance_score": balance_score,
            }
        )
    return pd.DataFrame(results)



def main():

    print("MIAMI-DADE (FL) & ORANGE (FL) SEVERE OUTAGE CLASSIFICATION")

    utils = ModelUtils()
    df, selected_features = utils.load_data_with_features()

    # ---- NEW: compute county medians after df is loaded ----
    print("\n--- COUNTY MEDIANS ---")
    county_medians = {}
    for county in [
        {"fips": 12086, "name": "Miami_Dade"},
        {"fips": 12095, "name": "Orange_FL"},
    ]:
        fips = county["fips"]
        median_val = df[df["fips_code"] == fips]["pct_out_area_unified"].median()
        county_medians[fips] = median_val
        print(f"{county['name']} median pct_out: {median_val:.6f}")

    md_median = county_medians[12086]
    orange_median = county_medians[12095]
    # --------------------------------------------------------

    target_counties = [
        {"fips": 12086, "name": "Miami_Dade_County_FL"},
        {"fips": 12095, "name": "Orange_County_FL"},
    ]

    # Thresholds to test (percentage of customers affected)
    test_thresholds = [
        0.0001,        # 0.01%
        md_median,     # Miami-Dade median
        orange_median, # Orange median
        0.0003,        # 0.03%
        0.0005,        # 0.05%
        0.001,         # 0.1% (industry-ish)
        0.005,         # 0.5%
        0.01,          # 1%
    ]
    
    all_results = {}
    results_dir = utils.results_dir / "county_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000, 2000],
        "class_weight": ["balanced"],
    }
    for county in target_counties:
        fips = county["fips"]
        name = county["name"]
        print(f"Processing {name} ({fips})")
        county_df = df[df["fips_code"] == fips].copy()
        if len(county_df) == 0:
            print(f"No data found for {name}!")
            continue

        print(f"Total samples: {len(county_df):,}")
        pct_col = "pct_out_area_unified"

        # Threshold Analysis
        print("\n--- THRESHOLD ANALYSIS ---")
        threshold_results = find_balanced_threshold(county_df[pct_col], test_thresholds)
        print(
            f"\n{'Threshold':>12} | {'Non-Severe':>12} | {'Severe':>12} | {'Severe %':>10}"
        )
        print("-" * 55)
        for _, row in threshold_results.iterrows():
            print(
                f"{row['threshold']:>12.4f} | {row['non_severe']:>12,} | {row['severe']:>12,} | {row['severe_ratio']:>10.2%}"
            )

        # Find best balanced threshold
        best_idx = threshold_results["balance_score"].idxmin()
        best_threshold = threshold_results.loc[best_idx, "threshold"]
        print(
            f"\nBest balanced threshold: {best_threshold:.4f} ({best_threshold*100:.2f}%)"
        )
        print(f"  Severe ratio: {threshold_results.loc[best_idx, 'severe_ratio']:.2%}")

        # Create binary target: 0 = Non-Severe, 1 = Severe
        county_df["severe_outage"] = (county_df[pct_col] >= best_threshold).astype(int)
        target_col = "severe_outage"

        # Split data: Test=2022, Train/Val=Others
        test_mask = county_df["year"] == 2022
        train_val_mask = ~test_mask
        test_df = county_df[test_mask].copy()
        train_val_df = county_df[train_val_mask].copy()
        if train_val_df[target_col].nunique() < 2:
            print("Error: Training data has only one class. Skipping.")
            continue

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.2,
            random_state=42,
            stratify=train_val_df[target_col],
        )
        print(f"\nTrain set: {len(train_df):,} samples")
        print(f"Val set:   {len(val_df):,} samples")
        print(f"Test set:  {len(test_df):,} samples (Year 2022)")

        X_train = train_df[selected_features]
        y_train = train_df[target_col]
        X_val = val_df[selected_features]
        y_val = val_df[target_col]
        X_test = test_df[selected_features]
        y_test = test_df[target_col]

        print("\nClass Distribution (Train Set):")
        print(
            f"  Non-Severe (0): {(y_train == 0).sum():,} ({(y_train == 0).mean():.2%})"
        )
        print(
            f"  Severe (1):     {(y_train == 1).sum():,} ({(y_train == 1).mean():.2%})"
        )
        print("\nClass Distribution (Test Set - 2022):")
        print(f"  Non-Severe (0): {(y_test == 0).sum():,} ({(y_test == 0).mean():.2%})")
        print(f"  Severe (1):     {(y_test == 1).sum():,} ({(y_test == 1).mean():.2%})")

        if y_train.nunique() < 2:
            print("Error: Training set has only one class. Cannot train classifier.")
            continue

        # Hyperparameter Tuning
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

        print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
        print("Best Hyperparameters:")
        for k, v in search.best_params_.items():
            print(f"  {k}: {v}")

        # Evaluation
        print("\n--- EVALUATION (2022 Test Set) ---")
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        if len(y_test.unique()) > 1:
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        else:
            roc_auc = 0.0
            pr_auc = 0.0
            print("Warning: Test set has only one class. ROC-AUC/PR-AUC set to 0.")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC:  {roc_auc:.4f}")
        print(f"PR AUC:   {pr_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=["Non-Severe", "Severe"], zero_division=0
            )
        )

        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  TN: {cm[0,0]:4} | FP: {cm[0,1]:4}")
        print(f"  FN: {cm[1,0]:4} | TP: {cm[1,1]:4}")
        # Save Results
        result = {
            "threshold_used": best_threshold,
            "threshold_pct": f"{best_threshold*100:.4f}%",
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            "test_metrics": {
                "accuracy": acc,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "f1": f1,
            },
            "test_class_distribution": {
                "non_severe": int((y_test == 0).sum()),
                "severe": int((y_test == 1).sum()),
            },
            "report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": cm.tolist(),
            "threshold_analysis": threshold_results.to_dict(orient="records"),
        }
        all_results[name] = result
        model_dir = utils.models_dir / "county_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_dir / f"{name}_severe_logistic_regression.pkl")
        preds_df = test_df[
            ["fips_code", "year", "month", "day_of_year", pct_col]
        ].copy()
        preds_df["actual"] = y_test
        preds_df["predicted"] = y_pred
        preds_df["probability"] = y_proba
        preds_df["threshold_used"] = best_threshold
        preds_df.to_csv(
            results_dir / f"{name}_severe_2022_predictions.csv", index=False
        )

    with open(results_dir / "miami_orange_severe_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("Processing Complete!")
    print(f"Models saved to: {utils.models_dir / 'county_models'}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
