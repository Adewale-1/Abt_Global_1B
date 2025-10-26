"""
Model Evaluation Script

I perform comprehensive evaluation of the trained logistic regression model
with focus on metrics appropriate for imbalanced classification.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.model_utils import ModelUtils


def calculate_business_metrics(y_true, y_pred, y_proba):
    """
    Calculate business-relevant metrics for power outage prediction.

    I focus on metrics that matter for operational decisions:
    - False Negative Rate: Missed outages (critical for safety)
    - False Positive Rate: False alarms (operational cost)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # calculate rates
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
    }


def format_confusion_matrix(cm, class_names):
    """
    Format confusion matrix for readable display.

    create both raw counts and percentage versions.
    """
    # calculate percentages
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    lines = [
        "Confusion Matrix (Raw Counts):",
        "-" * 60,
        f"{'':20} {'Predicted No-Outage':>20} {'Predicted Outage':>20}",
        f"{'Actual No-Outage':20} {cm[0,0]:>20,} {cm[0,1]:>20,}",
        f"{'Actual Outage':20} {cm[1,0]:>20,} {cm[1,1]:>20,}",
        "",
        "Confusion Matrix (Percentages by True Class):",
        "-" * 60,
        f"{'':20} {'Predicted No-Outage':>20} {'Predicted Outage':>20}",
        f"{'Actual No-Outage':20} {cm_percent[0,0]:>19.1f}% {cm_percent[0,1]:>19.1f}%",
        f"{'Actual Outage':20} {cm_percent[1,0]:>19.1f}% {cm_percent[1,1]:>19.1f}%",
    ]

    return "\n".join(lines)


def main():
    """
    Main evaluation function.
    """
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # initialize utilities
    utils = ModelUtils()

    # load the trained model
    print("STEP 1: LOADING MODEL")
    print("-" * 80)
    model, hyperparameters = utils.load_model("logistic_regression_best")

    print("\nModel Hyperparameters:")
    for param, value in hyperparameters.items():
        print(f"  {param}: {value}")

    # load data and prepare test set
    print("\nSTEP 2: LOADING TEST DATA")
    print("-" * 80)
    df, selected_features = utils.load_data_with_features()
    train_df, val_df, test_df = utils.temporal_split(df)

    X_test, y_test = utils.prepare_features_target(test_df, selected_features)

    print(f"\nTest set: {len(X_test):,} samples")
    print(f"  No-outage: {(y_test == 0).sum():,} ({(y_test == 0).mean():.2%})")
    print(f"  Outage: {(y_test == 1).sum():,} ({(y_test == 1).mean():.2%})")

    # generate predictions
    print("\nSTEP 3: GENERATING PREDICTIONS")
    print("-" * 80)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Predictions generated successfully")
    print(
        f"  Predicted no-outage: {(y_pred == 0).sum():,} ({(y_pred == 0).mean():.2%})"
    )
    print(f"  Predicted outage: {(y_pred == 1).sum():,} ({(y_pred == 1).mean():.2%})")

    # calculate comprehensive metrics
    print("\nSTEP 4: CALCULATING METRICS")
    print("-" * 80)

    # compute primary metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nPrimary Metrics:")
    print(f"  PR-AUC (Precision-Recall): {pr_auc:.4f} ** PRIMARY METRIC **")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    # calculate business metrics
    business_metrics = calculate_business_metrics(y_test, y_pred, y_proba)

    print(f"\nBusiness Metrics:")
    print(
        f"  False Negative Rate: {business_metrics['false_negative_rate']:.2%} (missed outages)"
    )
    print(
        f"  False Positive Rate: {business_metrics['false_positive_rate']:.2%} (false alarms)"
    )
    print(
        f"  True Positive Rate: {business_metrics['true_positive_rate']:.2%} (detected outages)"
    )
    print(
        f"  True Negative Rate: {business_metrics['true_negative_rate']:.2%} (correct no-outage)"
    )

    # generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_display = format_confusion_matrix(cm, ["No Outage", "Outage"])

    print("\n" + cm_display)

    # generate classification report
    print("\nDetailed Classification Report:")
    print("-" * 80)
    class_report = classification_report(
        y_test, y_pred, target_names=["No Outage", "Outage"], digits=4
    )
    print(class_report)

    # save predictions to CSV
    print("\nSTEP 5: SAVING PREDICTIONS")
    print("-" * 80)

    # use available columns (day was decomposed during encoding)
    available_cols = ["fips_code", "year", "month", "day_of_year"]
    metadata_cols = [col for col in available_cols if col in test_df.columns]

    predictions_df = test_df[metadata_cols].copy()
    predictions_df["actual"] = y_test.values
    predictions_df["predicted"] = y_pred
    predictions_df["probability_outage"] = y_proba
    predictions_df["correct"] = (
        predictions_df["actual"] == predictions_df["predicted"]
    ).astype(int)

    predictions_path = (
        utils.results_dir / "models" / "logistic_regression_predictions.csv"
    )
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")

    # create comprehensive evaluation report
    print("\nSTEP 6: SAVING EVALUATION REPORT")
    print("-" * 80)

    report_lines = [
        "=" * 80,
        "LOGISTIC REGRESSION MODEL EVALUATION REPORT",
        "=" * 80,
        f"Evaluated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Test Period: 2024 (year {test_df['year'].min()} to {test_df['year'].max()})",
        "",
        "MODEL INFORMATION",
        "-" * 80,
        "Model Type: Logistic Regression",
        f"Number of Features: {len(selected_features)}",
        "",
        "Hyperparameters:",
    ]

    for param, value in hyperparameters.items():
        report_lines.append(f"  {param}: {value}")

    report_lines.extend(
        [
            "",
            "TEST SET COMPOSITION",
            "-" * 80,
            f"Total Samples: {len(y_test):,}",
            f"  No-outage (Class 0): {(y_test == 0).sum():,} ({(y_test == 0).mean():.2%})",
            f"  Outage (Class 1): {(y_test == 1).sum():,} ({(y_test == 1).mean():.2%})",
            "",
            "PREDICTION DISTRIBUTION",
            "-" * 80,
            f"  Predicted no-outage: {(y_pred == 0).sum():,} ({(y_pred == 0).mean():.2%})",
            f"  Predicted outage: {(y_pred == 1).sum():,} ({(y_pred == 1).mean():.2%})",
            "",
            "PRIMARY METRICS (For Imbalanced Classification)",
            "-" * 80,
            f"Precision-Recall AUC: {pr_auc:.4f} ** PRIMARY METRIC **",
            f"  (Measures model's ability to distinguish classes in imbalanced data)",
            f"ROC-AUC: {roc_auc:.4f}",
            f"F1-Score: {f1:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall (Sensitivity): {recall:.4f}",
            "",
            "BUSINESS METRICS",
            "-" * 80,
            f"False Negative Rate: {business_metrics['false_negative_rate']:.2%}",
            f"  → Missed {business_metrics['false_negatives']:,} outages out of {business_metrics['false_negatives'] + business_metrics['true_positives']:,} actual outages",
            f"  → CRITICAL: These are outages the model failed to predict",
            "",
            f"False Positive Rate: {business_metrics['false_positive_rate']:.2%}",
            f"  → Incorrectly predicted {business_metrics['false_positives']:,} outages out of {business_metrics['false_positives'] + business_metrics['true_negatives']:,} no-outage days",
            f"  → COST: These result in unnecessary preventive actions",
            "",
            f"True Positive Rate (Recall): {business_metrics['true_positive_rate']:.2%}",
            f"  → Successfully detected {business_metrics['true_positives']:,} outages",
            "",
            f"True Negative Rate (Specificity): {business_metrics['true_negative_rate']:.2%}",
            f"  → Correctly identified {business_metrics['true_negatives']:,} no-outage days",
            "",
            cm_display,
            "",
            "DETAILED CLASSIFICATION REPORT",
            "-" * 80,
            class_report,
            "",
            "INTERPRETATION",
            "-" * 80,
            "The PR-AUC (Precision-Recall AUC) is the most important metric for this",
            "imbalanced dataset. It measures how well the model balances precision",
            "(avoiding false alarms) and recall (detecting actual outages).",
            "",
            f"With PR-AUC = {pr_auc:.4f}, the model shows {'strong' if pr_auc > 0.8 else 'moderate' if pr_auc > 0.6 else 'baseline'} performance.",
            "",
            "The false negative rate indicates the percentage of actual outages that",
            "were missed by the model. In a safety-critical application, minimizing",
            "this rate is crucial.",
            "",
            "NEXT STEPS",
            "-" * 80,
            "1. Review visualizations: python src/models/visualize_results.py",
            "2. Analyze feature coefficients to understand which weather patterns",
            "   most strongly predict outages",
            "3. Consider threshold tuning if business costs of FN vs FP differ",
            "4. Compare with other models (Random Forest, XGBoost) for improvement",
            "",
        ]
    )

    utils.save_results(
        results="\n".join(report_lines),
        filename="logistic_regression_evaluation.txt",
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  PR-AUC: {pr_auc:.4f} (primary metric)")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(
        f"  False Negative Rate: {business_metrics['false_negative_rate']:.2%} (missed outages)"
    )
    print(f"\nNext step:")
    print("  Generate visualizations: python src/models/visualize_results.py")


if __name__ == "__main__":
    main()
