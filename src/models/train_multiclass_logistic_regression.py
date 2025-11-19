"""
Multiclass Logistic Regression Model Training with Hyperparameter Tuning

I implement a complete multiclass training pipeline with:
- 3-class severity classification (No Outage, Minor, Severe)
- Temporal train/validation/test split
- Hyperparameter tuning using RandomizedSearchCV
- Class balancing to handle imbalance
- Comprehensive logging and model saving
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    log_loss,
    f1_score,
    accuracy_score,
)

from src.models.model_utils import ModelUtils, get_class_weights_info


def main():
    """
    Main training function with hyperparameter tuning.
    """
    print("=" * 80)
    print("MULTICLASS LOGISTIC REGRESSION MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    utils = ModelUtils()

    print("-" * 80)
    threshold = utils.load_best_threshold()
    print(f"Using threshold: {threshold:.0%}")

    print("-" * 80)
    df, selected_features = utils.load_data_with_features()

    print("\nCreating severity target...")
    print("-" * 80)
    df["outage_severity"] = utils.create_severity_target(df, threshold)

    print(f"\nSeverity class distribution (full dataset):")
    class_counts = df["outage_severity"].value_counts().sort_index()
    for class_label, class_name in [(0, "No Outage"), (1, "Minor"), (2, "Severe")]:
        count = class_counts.get(class_label, 0)
        pct = count / len(df)
        print(f"  {class_name} (Class {class_label}): {count:,} ({pct:.2%})")

    print("\nTemporal split...")
    print("-" * 80)
    train_df, val_df, test_df = utils.temporal_split(df, target_col="outage_severity")

    print("\nPreparing features and target...")
    print("-" * 80)
    X_train, y_train = utils.prepare_features_target(
        train_df, selected_features, target_col="outage_severity"
    )
    X_val, y_val = utils.prepare_features_target(
        val_df, selected_features, target_col="outage_severity"
    )
    X_test, y_test = utils.prepare_features_target(
        test_df, selected_features, target_col="outage_severity"
    )

    print("\nClass weight analysis...")
    print("-" * 80)
    weight_info = get_class_weights_info(y_train)
    print(f"Total samples: {weight_info['n_samples']:,}")
    print(f"Number of classes: {weight_info['n_classes']}")
    print("\nClass distribution:")
    for class_label, count in sorted(weight_info["class_counts"].items()):
        class_name = {0: "No Outage", 1: "Minor", 2: "Severe"}[class_label]
        print(
            f"  Class {class_label} ({class_name}): {count:,} samples ({count/weight_info['n_samples']:.2%})"
        )
    print("\nBalanced class weights (penalty multipliers):")
    for class_label, weight in sorted(weight_info["class_weights"].items()):
        class_name = {0: "No Outage", 1: "Minor", 2: "Severe"}[class_label]
        print(f"  Class {class_label} ({class_name}): {weight:.2f}x")

    print("\nHyperparameter tuning setup...")
    print("-" * 80)
    param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["saga", "lbfgs"],
        "multi_class": ["multinomial"],
        "class_weight": ["balanced"],
        "max_iter": [1000, 2000],
    }

    print("Parameter distributions:")
    for param, values in param_distributions.items():
        print(f"  {param}: {values}")

    print(f"\nTotal possible combinations: {6 * 2 * 2 * 1 * 1 * 2} = 48")
    print("Testing 20 random combinations with 5-fold cross-validation")
    print("Scoring metric: neg_log_loss (multiclass)")

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,
        cv=cv_strategy,
        scoring="neg_log_loss",
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True,
    )

    print("\nTraining with cross-validation...")
    print("-" * 80)
    print("This may take a few minutes...\n")

    search.fit(X_train, y_train)
    print("\nTraining completed!")
    print(f"Best cross-validation neg_log_loss: {search.best_score_:.4f}")

    print(f"\nBest hyperparameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    best_model = search.best_estimator_

    print("\nValidation set evaluation...")
    print("-" * 80)

    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)

    val_log_loss_score = log_loss(y_val, y_val_proba)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")
    val_weighted_f1 = f1_score(y_val, y_val_pred, average="weighted")

    print(f"Validation Log Loss: {val_log_loss_score:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Macro F1: {val_macro_f1:.4f}")
    print(f"Validation Weighted F1: {val_weighted_f1:.4f}")

    print("\nValidation Classification Report:")
    print(
        classification_report(
            y_val, y_val_pred, target_names=["No Outage", "Minor", "Severe"]
        )
    )

    print("\nPreliminary test set evaluation...")
    print("-" * 80)

    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)

    test_log_loss_score = log_loss(y_test, y_test_proba)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
    test_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")

    print(f"Test Log Loss: {test_log_loss_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}")

    print("\nTest Classification Report:")
    print(
        classification_report(
            y_test, y_test_pred, target_names=["No Outage", "Minor", "Severe"]
        )
    )

    print("\nSaving model and results...")
    print("-" * 80)

    training_info = {
        "model_type": "LogisticRegression (Multiclass)",
        "n_classes": 3,
        "class_names": ["No Outage", "Minor", "Severe"],
        "threshold_used": float(threshold),
        "n_features": len(selected_features),
        "feature_names": selected_features,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "class_distribution_train": weight_info["class_counts"],
        "class_weights": weight_info["class_weights"],
        "cv_strategy": "StratifiedKFold(n_splits=5)",
        "n_iter": 20,
        "best_cv_score": float(search.best_score_),
        "val_log_loss": float(val_log_loss_score),
        "val_accuracy": float(val_accuracy),
        "val_macro_f1": float(val_macro_f1),
        "val_weighted_f1": float(val_weighted_f1),
        "test_log_loss": float(test_log_loss_score),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
    }

    utils.save_model(
        model=best_model,
        model_name="multiclass_logistic_regression_best",
        hyperparameters=search.best_params_,
        training_info=training_info,
    )

    log_content = [
        "=" * 80,
        "MULTICLASS LOGISTIC REGRESSION TRAINING LOG",
        "=" * 80,
        f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "THRESHOLD INFORMATION",
        "-" * 80,
        f"Severity threshold: {threshold:.0%}",
        "",
        "HYPERPARAMETER TUNING RESULTS",
        "-" * 80,
        f"Total combinations tested: 20",
        f"Best CV neg_log_loss: {search.best_score_:.4f}",
        "",
        "Best Hyperparameters:",
    ]

    for param, value in search.best_params_.items():
        log_content.append(f"  {param}: {value}")

    log_content.extend(
        [
            "",
            "VALIDATION SET PERFORMANCE",
            "-" * 80,
            f"Log Loss: {val_log_loss_score:.4f}",
            f"Accuracy: {val_accuracy:.4f}",
            f"Macro F1: {val_macro_f1:.4f}",
            f"Weighted F1: {val_weighted_f1:.4f}",
            "",
            "TEST SET PERFORMANCE (Preliminary)",
            "-" * 80,
            f"Log Loss: {test_log_loss_score:.4f}",
            f"Accuracy: {test_accuracy:.4f}",
            f"Macro F1: {test_macro_f1:.4f}",
            f"Weighted F1: {test_weighted_f1:.4f}",
            "",
            "TOP 10 CV RESULTS",
            "-" * 80,
        ]
    )

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results_sorted = cv_results.sort_values("rank_test_score")

    for idx, row in cv_results_sorted.head(10).iterrows():
        log_content.append(
            f"Rank {int(row['rank_test_score'])}: "
            f"Mean={row['mean_test_score']:.4f}, "
            f"Std={row['std_test_score']:.4f}, "
            f"C={row['param_C']}, "
            f"penalty={row['param_penalty']}, "
            f"solver={row['param_solver']}, "
            f"max_iter={row['param_max_iter']}"
        )

    log_content.extend(
        [
            "",
            "FEATURE INFORMATION",
            "-" * 80,
            f"Number of features: {len(selected_features)}",
            f"Features: {', '.join(selected_features[:10])}...",
            "",
            "CLASS WEIGHT ANALYSIS",
            "-" * 80,
        ]
    )

    for class_label, weight in sorted(weight_info["class_weights"].items()):
        class_name = {0: "No Outage", 1: "Minor", 2: "Severe"}[class_label]
        count = weight_info["class_counts"][class_label]
        log_content.append(
            f"Class {class_label} ({class_name}): {count:,} samples, weight: {weight:.2f}x"
        )

    utils.save_results(
        results="\n".join(log_content),
        filename="multiclass_logistic_regression_training_log.txt",
    )

    print("\n")
    print("TRAINING COMPLETE!")



if __name__ == "__main__":
    main()
