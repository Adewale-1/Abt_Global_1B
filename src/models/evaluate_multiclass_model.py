"""
Multiclass Model Evaluation Script

I perform comprehensive evaluation of the trained multiclass logistic regression model
with focus on metrics appropriate for 3-class severity classification.
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
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.models.model_utils import ModelUtils


def format_confusion_matrix(cm, class_names):
    """
    Format confusion matrix for readable display.

    create both raw counts and percentage versions.
    """
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    lines = [
        "Confusion Matrix (Raw Counts):",
        "-" * 80,
        f"{'':20} {'Pred No':>12} {'Pred Minor':>12} {'Pred Severe':>12}",
    ]

    for i, class_name in enumerate(class_names):
        lines.append(
            f"{'Actual ' + class_name:20} "
            f"{cm[i, 0]:>12,} {cm[i, 1]:>12,} {cm[i, 2]:>12,}"
        )

    lines.extend(
        [
            "",
            "Confusion Matrix (Percentages by True Class):",
            "-" * 80,
            f"{'':20} {'Pred No':>12} {'Pred Minor':>12} {'Pred Severe':>12}",
        ]
    )

    for i, class_name in enumerate(class_names):
        lines.append(
            f"{'Actual ' + class_name:20} "
            f"{cm_percent[i, 0]:>11.1f}% {cm_percent[i, 1]:>11.1f}% {cm_percent[i, 2]:>11.1f}%"
        )

    return "\n".join(lines)


def calculate_per_class_metrics(y_true, y_pred, y_proba, class_names):
    """
    Calculate per-class metrics for multiclass classification.
    """
    n_classes = len(class_names)
    metrics = {}

    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        y_proba_binary = y_proba[:, i]

        tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()
        tn = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics[class_name] = {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(y_true_binary.sum()),
        }

    return metrics


def main():
    """
    Main evaluation function.
    """
    print("=" * 80)
    print("MULTICLASS MODEL EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    utils = ModelUtils()

    print("\nLOADING MODEL")
    print("-" * 80)
    model, hyperparameters = utils.load_model("multiclass_logistic_regression_best")

    print("\nModel Hyperparameters:")
    for param, value in hyperparameters.items():
        print(f"  {param}: {value}")

    print("\nLOADING TEST DATA")
    print("-" * 80)
    df, selected_features = utils.load_data_with_features()

    threshold = utils.load_best_threshold()
    df["outage_severity"] = utils.create_severity_target(df, threshold)

    train_df, val_df, test_df = utils.temporal_split(df, target_col="outage_severity")

    print(f"\nTest set: {len(test_df):,} samples")
    class_counts = test_df["outage_severity"].value_counts().sort_index()
    for class_label, class_name in [(0, "No Outage"), (1, "Minor"), (2, "Severe")]:
        count = class_counts.get(class_label, 0)
        pct = count / len(test_df)
        print(f"  {class_name}: {count:,} ({pct:.2%})")

    X_test, y_test = utils.prepare_features_target(
        test_df, selected_features, target_col="outage_severity"
    )

    print("\nGENERATING PREDICTIONS")
    print("-" * 80)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("Predictions generated successfully")
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    for class_label, class_name in [(0, "No Outage"), (1, "Minor"), (2, "Severe")]:
        count = pred_counts.get(class_label, 0)
        pct = count / len(y_pred)
        print(f"  Predicted {class_name}: {count:,} ({pct:.2%})")

    print("\nCALCULATING METRICS")
    print("-" * 80)

    accuracy = accuracy_score(y_test, y_pred)
    log_loss_score = log_loss(y_test, y_proba)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_precision = precision_score(y_test, y_pred, average="macro")
    weighted_precision = precision_score(y_test, y_pred, average="weighted")
    macro_recall = recall_score(y_test, y_pred, average="macro")
    weighted_recall = recall_score(y_test, y_pred, average="weighted")

    print("\nPrimary Metrics:")
    print(f"  Log Loss: {log_loss_score:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1-Score: {macro_f1:.4f}")
    print(f"  Weighted F1-Score: {weighted_f1:.4f}")
    print(f"  Macro Precision: {macro_precision:.4f}")
    print(f"  Weighted Precision: {weighted_precision:.4f}")
    print(f"  Macro Recall: {macro_recall:.4f}")
    print(f"  Weighted Recall: {weighted_recall:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    class_names = ["No Outage", "Minor", "Severe"]

    print(f"\n{format_confusion_matrix(cm, class_names)}")

    per_class_metrics = calculate_per_class_metrics(
        y_test, y_pred, y_proba, class_names
    )

    print("\nPer-Class Metrics:")
    print("-" * 80)
    for class_name, metrics in per_class_metrics.items():
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Support: {metrics['support']:,}")
        print(f"  True Positives: {metrics['true_positives']:,}")
        print(f"  False Positives: {metrics['false_positives']:,}")
        print(f"  False Negatives: {metrics['false_negatives']:,}")

    print("\nDetailed Classification Report:")
    print("-" * 80)
    class_report = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )
    print(class_report)

    print("\nSAVING PREDICTIONS")
    print("-" * 80)

    available_cols = ["fips_code", "year", "month", "day_of_year"]
    metadata_cols = [col for col in available_cols if col in test_df.columns]

    predictions_df = test_df[metadata_cols].copy()
    predictions_df["actual"] = y_test.values
    predictions_df["predicted"] = y_pred
    predictions_df["prob_no_outage"] = y_proba[:, 0]
    predictions_df["prob_minor"] = y_proba[:, 1]
    predictions_df["prob_severe"] = y_proba[:, 2]
    predictions_df["correct"] = (
        predictions_df["actual"] == predictions_df["predicted"]
    ).astype(int)

    predictions_path = utils.results_dir / "models" / "multiclass_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    print("\nSAVING EVALUATION REPORT")
    print("-" * 80)

    evaluation_report = [
        "=" * 80,
        "MULTICLASS LOGISTIC REGRESSION EVALUATION REPORT",
        "=" * 80,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "MODEL INFORMATION",
        "-" * 80,
        f"Model: Multiclass Logistic Regression",
        f"Threshold Used: {threshold:.0%}",
        f"Number of Features: {len(selected_features)}",
        "",
        "TEST SET INFORMATION",
        "-" * 80,
        f"Total Samples: {len(test_df):,}",
    ]

    for class_label, class_name in [(0, "No Outage"), (1, "Minor"), (2, "Severe")]:
        count = class_counts.get(class_label, 0)
        pct = count / len(test_df)
        evaluation_report.append(f"{class_name}: {count:,} ({pct:.2%})")

    evaluation_report.extend(
        [
            "",
            "PRIMARY METRICS",
            "-" * 80,
            f"Log Loss: {log_loss_score:.4f}",
            f"Accuracy: {accuracy:.4f}",
            f"Macro F1-Score: {macro_f1:.4f}",
            f"Weighted F1-Score: {weighted_f1:.4f}",
            f"Macro Precision: {macro_precision:.4f}",
            f"Weighted Precision: {weighted_precision:.4f}",
            f"Macro Recall: {macro_recall:.4f}",
            f"Weighted Recall: {weighted_recall:.4f}",
            "",
            format_confusion_matrix(cm, class_names),
            "",
            "PER-CLASS METRICS",
            "-" * 80,
        ]
    )

    for class_name, metrics in per_class_metrics.items():
        evaluation_report.extend(
            [
                f"\n{class_name}:",
                f"  Precision: {metrics['precision']:.4f}",
                f"  Recall: {metrics['recall']:.4f}",
                f"  F1-Score: {metrics['f1_score']:.4f}",
                f"  Support: {metrics['support']:,}",
                f"  True Positives: {metrics['true_positives']:,}",
                f"  False Positives: {metrics['false_positives']:,}",
                f"  False Negatives: {metrics['false_negatives']:,}",
            ]
        )

    evaluation_report.extend(
        [
            "",
            "DETAILED CLASSIFICATION REPORT",
            "-" * 80,
            class_report,
        ]
    )

    utils.save_results(
        results="\n".join(evaluation_report),
        filename="multiclass_evaluation.txt",
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - {predictions_path}")
    print(f"  - {utils.results_dir / 'models' / 'multiclass_evaluation.txt'}")



if __name__ == "__main__":
    main()
