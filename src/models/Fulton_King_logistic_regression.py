import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
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
from pathlib import Path
import json
from datetime import datetime


def main():
    print("FULTON (GA) & KING (WA) LOGISTIC REGRESSION TRAINING")

    utils = ModelUtils()
    df, selected_features = utils.load_data_with_features()
    target_counties = [
        {"fips": 13121, "name": "Fulton_County_GA"},
        {"fips": 53033, "name": "King_County_WA"},
    ]
    all_results = {}
    results_dir = utils.results_dir / "county_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Hyperparameter search space (same as train_logistic_regression)
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

        print(f"\n{'='*40}")
        print(f"Processing {name} ({fips})")
        print(f"{'='*40}")

        county_df = df[df["fips_code"] == fips].copy()
        if len(county_df) == 0:
            print(f"No data found for {name}!")
            continue

        print(f"Total samples: {len(county_df):,}")

        test_mask = county_df["year"] == 2022
        train_val_mask = ~test_mask
        test_df = county_df[test_mask].copy()
        train_val_df = county_df[train_val_mask].copy()

        # Split Train/Val (80/20 split of the non-2022 data)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.2,
            random_state=42,
            stratify=(
                train_val_df["any_out"]
                if train_val_df["any_out"].nunique() > 1
                else None
            ),
        )

        print(f"Train set: {len(train_df):,} samples")
        print(f"Val set:   {len(val_df):,} samples")
        print(f"Test set:  {len(test_df):,} samples (Year 2022)")

        # Prepare features and target
        target_col = "any_out"
        X_train = train_df[selected_features]
        y_train = train_df[target_col]
        X_val = val_df[selected_features]
        y_val = val_df[target_col]
        X_test = test_df[selected_features]
        y_test = test_df[target_col]

        print("\nClass Weight Analysis (Train Set):")
        weight_info = get_class_weights_info(y_train)
        for cls, weight in weight_info["class_weights"].items():
            print(f"  Class {cls}: {weight:.2f}x")

        # Check if we have both classes in train
        if len(y_train.unique()) < 2:
            print("Error: Training set has only one class. Cannot train classifier.")
            continue

        # Hyperparameter Tuning
        print("\nStarting Hyperparameter Tuning (RandomizedSearchCV)...")
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

        # Evaluation
        print("\nEvaluating on 2022 Test Set...")
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Metrics
        # Handle case where test set has only one class (common in small county-years)
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
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        # Save Results
        result = {
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            "test_metrics": {
                "accuracy": acc,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "f1": f1,
            },
            "report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        all_results[name] = result
        # Save Model
        model_dir = utils.models_dir / "county_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_dir / f"{name}_logistic_regression.pkl")
        preds_df = test_df[["fips_code", "year", "month", "day_of_year"]].copy()
        preds_df["actual"] = y_test
        preds_df["predicted"] = y_pred
        preds_df["probability"] = y_proba
        preds_df.to_csv(results_dir / f"{name}_2022_predictions.csv", index=False)

    # Save Summary
    with open(results_dir / "fulton_king_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("Processing Complete!")
    print(f"Models saved to: {utils.models_dir / 'county_models'}")
    print(f"Results saved to: {utils.results_dir / 'county_models'}")


if __name__ == "__main__":
    main()
