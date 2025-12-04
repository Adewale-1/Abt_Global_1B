import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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

# 🚫 Silence ALL warnings globally
import warnings
warnings.filterwarnings("ignore")



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


def evaluate_classifier(model, X_test, y_test, model_name="model"):
    """Run standard evaluation metrics for a binary classifier."""
    y_pred = model.predict(X_test)

    # Some classifiers (like HGB) may not implement predict_proba in older versions;
    # fall back to decision_function if needed.
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        # Fallback: use predictions as scores (not ideal, but avoids crashing)
        y_scores = y_pred

    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
    else:
        roc_auc = 0.0
        pr_auc = 0.0
        print(f"Warning: Test set has only one class for {model_name}. ROC-AUC/PR-AUC set to 0.")

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(
        y_test, y_pred, target_names=["Non-Severe", "Severe"], zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    # For console printing (optional)
    print(f"\n--- EVALUATION ({model_name}) ---")
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
    print(f"  TN: {cm[0, 0]:4} | FP: {cm[0, 1]:4}")
    print(f"  FN: {cm[1, 0]:4} | TP: {cm[1, 1]:4}")

    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def main():

    print("MIAMI-DADE (FL) & ORANGE (FL) SEVERE OUTAGE CLASSIFICATION")

    utils = ModelUtils()
    df, selected_features = utils.load_data_with_features()

    # ---- Compute county medians after df is loaded ----
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
        0.001,         # 0.1%
        0.005,         # 0.5%
        0.01,          # 1%
    ]

    all_results = {}
    results_dir = utils.results_dir / "county_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = utils.models_dir / "county_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameter grids
    logit_param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000, 2000],
        "class_weight": ["balanced"],
    }

    hgb_param_distributions = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, None],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 50],
    }

    rf_param_distributions = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for county in target_counties:
        fips = county["fips"]
        name = county["name"]
        print(f"\n==============================")
        print(f"Processing {name} ({fips})")
        print(f"==============================")

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
            print("Error: Training set has only one class. Cannot train classifiers.")
            continue

        county_result = {
            "threshold_used": float(best_threshold),
            "threshold_pct": f"{best_threshold*100:.4f}%",
            "threshold_analysis": threshold_results.to_dict(orient="records"),
        }

        # ------------------------------------------------------------------
        # 1) LOGISTIC REGRESSION
        # ------------------------------------------------------------------
        print("\n--- HYPERPARAMETER TUNING: LOGISTIC REGRESSION ---")
        logit_search = RandomizedSearchCV(
            estimator=LogisticRegression(random_state=42),
            param_distributions=logit_param_distributions,
            n_iter=20,
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True,
        )
        logit_search.fit(X_train, y_train)
        best_logit = logit_search.best_estimator_

        print(f"\nBest CV ROC-AUC (Logit): {logit_search.best_score_:.4f}")
        print("Best Hyperparameters (Logit):")
        for k, v in logit_search.best_params_.items():
            print(f"  {k}: {v}")

        logit_eval = evaluate_classifier(
            best_logit, X_test, y_test, model_name="2022 Test Set - Logistic Regression"
        )

        county_result["logistic_regression"] = {
            "best_params": logit_search.best_params_,
            "best_cv_score": float(logit_search.best_score_),
            "test_metrics": {
                "accuracy": logit_eval["accuracy"],
                "roc_auc": logit_eval["roc_auc"],
                "pr_auc": logit_eval["pr_auc"],
                "f1": logit_eval["f1"],
            },
            "test_class_distribution": {
                "non_severe": int((y_test == 0).sum()),
                "severe": int((y_test == 1).sum()),
            },
            "report": logit_eval["report"],
            "confusion_matrix": logit_eval["confusion_matrix"],
        }

        # Save logistic model
        joblib.dump(best_logit, model_dir / f"{name}_severe_logistic_regression.pkl")

        # Save logistic test predictions CSV (as before)
        if hasattr(best_logit, "predict_proba"):
            y_proba_logit = best_logit.predict_proba(X_test)[:, 1]
        else:
            y_proba_logit = best_logit.decision_function(X_test)

        preds_df = test_df[
            ["fips_code", "year", "month", "day_of_year", pct_col]
        ].copy()
        preds_df["actual"] = y_test
        preds_df["predicted"] = best_logit.predict(X_test)
        preds_df["probability"] = y_proba_logit
        preds_df["threshold_used"] = best_threshold
        preds_df.to_csv(
            results_dir / f"{name}_severe_2022_predictions_logistic.csv", index=False
        )

        # ------------------------------------------------------------------
        # 2) HIST GRADIENT BOOSTING (TREE ENSEMBLE)
        # ------------------------------------------------------------------
        print("\n--- HYPERPARAMETER TUNING: HIST GRADIENT BOOSTING ---")
        hgb_search = RandomizedSearchCV(
            estimator=HistGradientBoostingClassifier(random_state=42),
            param_distributions=hgb_param_distributions,
            n_iter=15,
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True,
        )
        hgb_search.fit(X_train, y_train)
        best_hgb = hgb_search.best_estimator_

        print(f"\nBest CV ROC-AUC (HGB): {hgb_search.best_score_:.4f}")
        print("Best Hyperparameters (HGB):")
        for k, v in hgb_search.best_params_.items():
            print(f"  {k}: {v}")

        hgb_eval = evaluate_classifier(
            best_hgb, X_test, y_test, model_name="2022 Test Set - HistGradientBoosting"
        )

        county_result["hist_gradient_boosting"] = {
            "best_params": hgb_search.best_params_,
            "best_cv_score": float(hgb_search.best_score_),
            "test_metrics": {
                "accuracy": hgb_eval["accuracy"],
                "roc_auc": hgb_eval["roc_auc"],
                "pr_auc": hgb_eval["pr_auc"],
                "f1": hgb_eval["f1"],
            },
            "test_class_distribution": {
                "non_severe": int((y_test == 0).sum()),
                "severe": int((y_test == 1).sum()),
            },
            "report": hgb_eval["report"],
            "confusion_matrix": hgb_eval["confusion_matrix"],
        }

        # Save HGB model
        joblib.dump(best_hgb, model_dir / f"{name}_severe_histgb_classifier.pkl")

        # ------------------------------------------------------------------
        # 3) RANDOM FOREST
        # ------------------------------------------------------------------
        print("\n--- HYPERPARAMETER TUNING: RANDOM FOREST ---")
        rf_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=rf_param_distributions,
            n_iter=15,
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True,
        )
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_

        print(f"\nBest CV ROC-AUC (RF): {rf_search.best_score_:.4f}")
        print("Best Hyperparameters (RF):")
        for k, v in rf_search.best_params_.items():
            print(f"  {k}: {v}")

        rf_eval = evaluate_classifier(
            best_rf, X_test, y_test, model_name="2022 Test Set - RandomForest"
        )

        county_result["random_forest"] = {
            "best_params": rf_search.best_params_,
            "best_cv_score": float(rf_search.best_score_),
            "test_metrics": {
                "accuracy": rf_eval["accuracy"],
                "roc_auc": rf_eval["roc_auc"],
                "pr_auc": rf_eval["pr_auc"],
                "f1": rf_eval["f1"],
            },
            "test_class_distribution": {
                "non_severe": int((y_test == 0).sum()),
                "severe": int((y_test == 1).sum()),
            },
            "report": rf_eval["report"],
            "confusion_matrix": rf_eval["confusion_matrix"],
        }

        # Save RF model
        joblib.dump(best_rf, model_dir / f"{name}_severe_random_forest.pkl")

        # ------------------------------------------------------------------
        # Save per-county results
        # ------------------------------------------------------------------
        all_results[name] = county_result

    # Save summary JSON across counties
    with open(results_dir / "miami_orange_severe_models_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nProcessing Complete!")
    print(f"Models saved to: {model_dir}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
