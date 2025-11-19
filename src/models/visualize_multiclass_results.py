"""
Multiclass Model Visualization Script

I generate comprehensive visualizations for multiclass model interpretation.
These plots help understand model performance and feature importance across
3 severity classes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

from src.models.model_utils import ModelUtils

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Create confusion matrix heatmap for 3-class classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    class_names = ["No Outage", "Minor", "Severe"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        ax=ax1,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=11)
    ax1.set_xlabel("Predicted", fontsize=11)

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        cbar=True,
        square=True,
        ax=ax2,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax2.set_title("Confusion Matrix (% by True Class)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Actual", fontsize=11)
    ax2.set_xlabel("Predicted", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")


def plot_class_distribution(y_true, y_pred, save_path):
    """
    Compare actual vs predicted class distributions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    class_names = ["No Outage", "Minor", "Severe"]

    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    true_pcts = true_counts / len(y_true) * 100
    pred_pcts = pred_counts / len(y_pred) * 100

    x = np.arange(len(class_names))
    width = 0.35

    ax1.bar(x - width/2, true_counts.values, width, label='Actual', color='steelblue')
    ax1.bar(x + width/2, pred_counts.values, width, label='Predicted', color='coral')
    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Class Distribution Comparison (Counts)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x - width/2, true_pcts.values, width, label='Actual', color='steelblue')
    ax2.bar(x + width/2, pred_pcts.values, width, label='Predicted', color='coral')
    ax2.set_xlabel('Class', fontsize=11)
    ax2.set_ylabel('Percentage', fontsize=11)
    ax2.set_title('Class Distribution Comparison (Percentages)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved class distribution to: {save_path}")


def plot_feature_coefficients(model, feature_names, save_path, top_n=20):
    """
    Plot feature coefficients heatmap for all 3 classes.
    """
    coefficients = model.coef_.T
    
    feature_importance = np.abs(coefficients).mean(axis=1)
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    
    top_features = [feature_names[i] for i in top_indices]
    top_coefs = coefficients[top_indices]
    
    coef_df = pd.DataFrame(
        top_coefs,
        index=top_features,
        columns=["No Outage", "Minor", "Severe"]
    )
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    
    sns.heatmap(
        coef_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Coefficient Value"},
        ax=ax,
        linewidths=0.5,
    )
    
    ax.set_title(
        f"Top {top_n} Feature Coefficients by Class", 
        fontsize=12, 
        fontweight="bold"
    )
    ax.set_xlabel("Severity Class", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved feature coefficients to: {save_path}")


def plot_prediction_confidence(y_true, y_proba, save_path):
    """
    Plot histogram of prediction confidence (max probability) by actual class.
    """
    max_proba = np.max(y_proba, axis=1)
    predicted_class = np.argmax(y_proba, axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    class_names = ["No Outage", "Minor", "Severe"]
    colors = ['steelblue', 'coral', 'forestgreen']
    
    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        ax1.hist(
            max_proba[mask],
            bins=30,
            alpha=0.6,
            label=f'{class_name} (Actual)',
            color=colors[class_idx]
        )
    
    ax1.set_xlabel('Max Probability (Prediction Confidence)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Prediction Confidence by Actual Class', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    correct_mask = (y_true == predicted_class)
    ax2.hist(
        max_proba[correct_mask],
        bins=30,
        alpha=0.6,
        label='Correct Predictions',
        color='forestgreen'
    )
    ax2.hist(
        max_proba[~correct_mask],
        bins=30,
        alpha=0.6,
        label='Incorrect Predictions',
        color='crimson'
    )
    
    ax2.set_xlabel('Max Probability (Prediction Confidence)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Prediction Confidence: Correct vs Incorrect', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved prediction confidence plot to: {save_path}")


def plot_roc_curves(y_true, y_proba, save_path):
    """
    Plot ROC curves for each class (one-vs-rest).
    """
    n_classes = 3
    class_names = ["No Outage", "Minor", "Severe"]
    colors = ['steelblue', 'coral', 'forestgreen']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for class_idx, class_name in enumerate(class_names):
        y_binary = (y_true == class_idx).astype(int)
        y_proba_binary = y_proba[:, class_idx]
        
        fpr, tpr, _ = roc_curve(y_binary, y_proba_binary)
        auc = roc_auc_score(y_binary, y_proba_binary)
        
        ax.plot(
            fpr,
            tpr,
            color=colors[class_idx],
            lw=2,
            label=f'{class_name} (AUC = {auc:.3f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved ROC curves to: {save_path}")


def main():
    """
    Main visualization function.
    """
    print("=" * 80)
    print("MULTICLASS MODEL VISUALIZATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    utils = ModelUtils()

    print("\nLOADING MODEL AND DATA")
    print("-" * 80)
    
    model, hyperparameters = utils.load_model("multiclass_logistic_regression_best")
    
    df, selected_features = utils.load_data_with_features()
    threshold = utils.load_best_threshold()
    df['outage_severity'] = utils.create_severity_target(df, threshold)
    
    _, _, test_df = utils.temporal_split(df, target_col="outage_severity")
    
    X_test, y_test = utils.prepare_features_target(
        test_df, selected_features, target_col="outage_severity"
    )

    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\nGENERATING VISUALIZATIONS")
    print("-" * 80)

    output_dir = utils.results_dir / "figures" / "multiclass"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCONFUSION MATRIX...")
    plot_confusion_matrix(
        y_test, y_pred, output_dir / "confusion_matrix.png"
    )

    print("\nCLASS DISTRIBUTION COMPARISON...")
    plot_class_distribution(
        y_test, y_pred, output_dir / "class_distribution.png"
    )

    print("\nFEATURE COEFFICIENTS HEATMAP...")
    plot_feature_coefficients(
        model, selected_features, output_dir / "feature_coefficients.png", top_n=20
    )

    print("\nPREDICTION CONFIDENCE DISTRIBUTION...")
    plot_prediction_confidence(
        y_test, y_proba, output_dir / "prediction_confidence.png"
    )

    print("\nROC CURVES (ONE-VS-REST)...")
    plot_roc_curves(
        y_test, y_proba, output_dir / "roc_curves.png"
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - class_distribution.png")
    print("  - feature_coefficients.png")
    print("  - prediction_confidence.png")
    print("  - roc_curves.png")


if __name__ == "__main__":
    main()

