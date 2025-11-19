"""
Logistic Regression Model Training with Hyperparameter Tuning

I implement a complete training pipeline with:
- Temporal train/validation/test split
- Hyperparameter tuning using RandomizedSearchCV
- Class balancing to handle 95.64% imbalance
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
    roc_auc_score,
    average_precision_score,
    make_scorer,
)

from src.models.model_utils import ModelUtils, get_class_weights_info


def main():
    """
    Main training function with hyperparameter tuning.
    """
    print("=" * 80)
    print("LOGISTIC REGRESSION MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    utils = ModelUtils()

    # load data and selected features
    print("LOADING DATA")
    print("-" * 80)
    df, selected_features = utils.load_data_with_features()

    # perform temporal split
    print("\nTEMPORAL SPLIT")
    print("-" * 80)
    train_df, val_df, test_df = utils.temporal_split(df)

    # prepare features and target for each set
    X_train, y_train = utils.prepare_features_target(train_df, selected_features)
    X_val, y_val = utils.prepare_features_target(val_df, selected_features)
    X_test, y_test = utils.prepare_features_target(test_df, selected_features)

    # analyze class weights to understand the imbalance correction
    print("\n CLASS WEIGHT ANALYSIS")
    print("-" * 80)
    weight_info = get_class_weights_info(y_train)
    print(f"Total samples: {weight_info['n_samples']:,}")
    print(f"Number of classes: {weight_info['n_classes']}")
    print("\nClass distribution:")
    for class_label, count in weight_info["class_counts"].items():
        print(
            f"  Class {class_label}: {count:,} samples ({count/weight_info['n_samples']:.2%})"
        )
    print("\nBalanced class weights (penalty multipliers):")
    for class_label, weight in weight_info["class_weights"].items():
        print(f"  Class {class_label}: {weight:.2f}x")
    print(
        f"\nThis means the model will penalize mistakes on class 0 (no-outage) "
        f"{weight_info['class_weights'][0]:.1f}x more than class 1 (outage)"
    )

    # define hyperparameter search space
    print("\nHYPERPARAMETER TUNING SETUP")
    print("-" * 80)
    param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength (inverse)
        "penalty": ["l1", "l2"],  # Regularization type
        "solver": ["liblinear", "saga"],  # Optimizers that support both l1 and l2
        "max_iter": [1000, 2000],  # Maximum iterations for convergence
        "class_weight": ["balanced"],  # Fixed to handle imbalance
    }

    print("Parameter distributions:")
    for param, values in param_distributions.items():
        print(f"  {param}: {values}")

    print(f"\nTotal possible combinations: {6 * 2 * 2 * 2 * 1} = 48")
    print("Testing 20 random combinations with 5-fold cross-validation")
    print("Scoring metric: ROC-AUC (area under ROC curve)")

    # configure RandomizedSearchCV
    # Using stratified k-fold to maintain class distribution in each fold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=LogisticRegression(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,  # Test 20 random combinations
        cv=cv_strategy,
        scoring="roc_auc",  # Primary metric for imbalanced classification
        n_jobs=-1,  # Use all available cores
        verbose=2,  # Show progress
        random_state=42,
        return_train_score=True,
    )

    # perform hyperparameter search
    print("\nTRAINING WITH CROSS-VALIDATION")
    print("-" * 80)
    print("This may take a few minutes...\n")

    search.fit(X_train, y_train)

    print("\nTraining completed!")
    print(f"Best cross-validation ROC-AUC: {search.best_score_:.4f}")
    print(f"\nBest hyperparameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    # get the best model
    best_model = search.best_estimator_

    # evaluate on validation set
    print("\nVALIDATION SET EVALUATION")
    print("-" * 80)

    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]

    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    val_pr_auc = average_precision_score(y_val, y_val_proba)

    print(f"Validation ROC-AUC: {val_roc_auc:.4f}")
    print(f"Validation PR-AUC: {val_pr_auc:.4f} (primary metric for imbalanced data)")

    print("\nValidation Classification Report:")
    print(
        classification_report(y_val, y_val_pred, target_names=["No Outage", "Outage"])
    )

    # also evaluate on test set for initial assessment
    print("\nPRELIMINARY TEST SET EVALUATION")
    print("-" * 80)

    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    print("\nTest Classification Report:")
    print(
        classification_report(y_test, y_test_pred, target_names=["No Outage", "Outage"])
    )

    # save the best model
    print("\nSAVING MODEL AND RESULTS")
    print("-" * 80)

    training_info = {
        "model_type": "LogisticRegression",
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
        "val_roc_auc": float(val_roc_auc),
        "val_pr_auc": float(val_pr_auc),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
    }

    utils.save_model(
        model=best_model,
        model_name="logistic_regression_best",
        hyperparameters=search.best_params_,
        training_info=training_info,
    )

    # save detailed training log
    log_content = [
        "=" * 80,
        "LOGISTIC REGRESSION TRAINING LOG",
        "=" * 80,
        f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "HYPERPARAMETER TUNING RESULTS",
        "-" * 80,
        f"Total combinations tested: 20",
        f"Best CV ROC-AUC: {search.best_score_:.4f}",
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
            f"ROC-AUC: {val_roc_auc:.4f}",
            f"PR-AUC: {val_pr_auc:.4f}",
            "",
            "TEST SET PERFORMANCE (Preliminary)",
            "-" * 80,
            f"ROC-AUC: {test_roc_auc:.4f}",
            f"PR-AUC: {test_pr_auc:.4f}",
            "",
            "TOP 10 CV RESULTS",
            "-" * 80,
        ]
    )

    # I extract and sort CV results
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results_sorted = cv_results.sort_values("rank_test_score")

    for idx, row in cv_results_sorted.head(10).iterrows():
        log_content.append(
            f"Rank {int(row['rank_test_score'])}: "
            f"Mean={row['mean_test_score']:.4f}, "
            f"Std={row['std_test_score']:.4f}, "
            f"C={row['param_C']}, "
            f"penalty={row['param_penalty']}, "
            f"solver={row['param_solver']}"
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
            f"Class 0 (no-outage) weight: {weight_info['class_weights'][0]:.2f}x",
            f"Class 1 (outage) weight: {weight_info['class_weights'][1]:.2f}x",
            "",
            "This means the model penalizes errors on the minority class (no-outage)",
            f"{weight_info['class_weights'][0]:.1f}x more than the majority class (outage).",
            "",
        ]
    )

    utils.save_results(
        results="\n".join(log_content),
        filename="logistic_regression_training_log.txt",
    )

    print("\n")
    print("TRAINING COMPLETE!")
    print(f"\nNext steps:")
    print("  Run evaluation: python src/models/evaluate_model.py")
    print("  Generate visualizations: python src/models/visualize_results.py")


if __name__ == "__main__":
    main()
