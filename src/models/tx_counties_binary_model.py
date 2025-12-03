import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.models.model_utils import ModelUtils, get_class_weights_info


def main():
    print("HARRIS (TX) & BEXAR (TX) LOGISTIC REGRESSION TRAINING")

    # ---------- 1. Load data + selected features ----------
    utils = ModelUtils()
    df, selected_features = utils.load_data_with_features()

    # We expect at least these columns to be present:
    # 'fips_code', 'year', 'any_out', plus selected_features.

    target_counties = [
        {"fips": 48201, "name": "Harris_County_TX"},
        {"fips": 48029, "name": "Bexar_County_TX"},
    ]

    all_results = {}
    results_dir = utils.results_dir / "county_models_tx"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 2. Hyperparameter search space ----------
    param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000, 2000],
        "class_weight": ["balanced"],
    }

    # ---------- 3. Loop over counties ----------
    for county in target_counties:
        fips = county["fips"]
        name = county["name"]

        print("\n" + "=" * 40)
        print(f"Processing {name} ({fips})")
        print("=" * 40)

        county_df = df[df["fips_code"] == fips].copy()
        if len(county_df) == 0:
            print(f"No data found for {name}!")
            continue

        print(f"Total samples: {len(county_df):,}")

        # ---------- 4. Temporal split ----------
        # Test set: year == 2022
        # Train/Val: all other years (2019–2021, 2023–2024)
        if "year" not in county_df.columns:
            raise ValueError("Expected a 'year' column in the dataset.")

        test_mask = county_df["year"] == 2022
        train_val_mask = ~test_mask

        test_df = county_df[test_mask].copy()
        train_val_df = county_df[train_val_mask].copy()

        if len(test_df) == 0:
            print("Warning: No 2022 rows found for test set.")
        if len(train_val_df) == 0:
            print("Error: No non-2022 rows for train/val; cannot train model.")
            continue

        print(f"Train+Val pool: {len(train_val_df):,} samples")
        print(f"Test set (2022): {len(test_df):,} samples")

        # ---------- 5. Train/Val split (80/20) ----------
        target_col = "any_out"

        if target_col not in train_val_df.columns:
            raise ValueError(
                f"Expected target column '{target_col}' in the dataset."
            )

        # 80/20 split of non-2022 data
        stratify_labels = (
            train_val_df[target_col]
            if train_val_df[target_col].nunique() > 1
            else None
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.2,
            random_state=42,
            stratify=stratify_labels,
        )

        print(f"Train set: {len(train_df):,} samples")
        print(f"Val set:   {len(val_df):,} samples")
        print(f"Test set:  {len(test_df):,} samples (Year 2022)")

        # ---------- 6. Feature matrices & labels ----------
        X_train = train_df[selected_features].fillna(0.0)
        y_train = train_df[target_col].astype(int)

        X_val = val_df[selected_features].fillna(0.0)
        y_val = val_df[target_col].astype(int)

        X_test = test_df[selected_features].fillna(0.0)
        y_test = test_df[target_col].astype(int)

        print("\nClass Weight Analysis (Train Set):")
        weight_info = get_class_weights_info(y_train)
        for cls, weight in weight_info["class_weights"].items():
            print(f"  Class {cls}: {weight:.2f}x")

        # Edge case: only one class in train
        if len(y_train.unique()) < 2:
            print(
                "Error: Training set has only one class. "
                "Cannot train classifier for this county."
            )
            continue

        # ---------- 7. Hyperparameter tuning ----------
        print("\nStarting Hyperparameter Tuning (RandomizedSearchCV)...")

        cv_strategy = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        )

        search = RandomizedSearchCV(
            estimator=LogisticRegression(random_state=42),
            param_distributions=param_distributions,
            n_iter=20,
            cv=cv_strategy,
            scoring="roc_auc",  # Optimize for ROC-AUC
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print("Best Hyperparameters:")
        for k, v in search.best_params_.items():
            print(f"  {k}: {v}")

        # ---------- 8. Evaluation on 2022 test set ----------
        print("\nEvaluating on 2022 Test Set...")
        y_pred = best_model.predict(X_test)
        # Some solvers don't expose predict_proba for all configurations; guard it.
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]
        else:
            # Fallback: use decision_function and squash to [0,1] via ranking
            scores = best_model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        if len(y_test.unique()) > 1:
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        else:
            roc_auc = 0.0
            pr_auc = 0.0
            print(
                "Warning: Test set has only one class. "
                "ROC-AUC and PR-AUC set to 0."
            )

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR  AUC: {pr_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # ---------- 9. Save results ----------
        result = {
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            "test_metrics": {
                "accuracy": acc,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
            },
            "report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        all_results[name] = result

        # Save model
        model_dir = utils.models_dir / "county_models_tx"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{name}_logistic_regression.pkl"
        joblib.dump(best_model, model_path)

        # Save test predictions for inspection
        preds_df = test_df[["fips_code", "year", "month", "day_of_year"]].copy()
        preds_df["actual"] = y_test.values
        preds_df["predicted"] = y_pred
        preds_df["probability"] = y_proba
        preds_path = results_dir / f"{name}_2022_predictions.csv"
        preds_df.to_csv(preds_path, index=False)

        print(f"\nSaved model to:     {model_path}")
        print(f"Saved predictions to: {preds_path}")

    # ---------- 10. Save summary over both counties ----------
    summary_path = results_dir / "tx_harris_bexar_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nProcessing Complete!")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
