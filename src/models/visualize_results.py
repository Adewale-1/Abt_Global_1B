"""
Model Visualization Script

I generate comprehensive visualizations for model interpretation and results
communication. These plots help stakeholders understand model performance and
which weather patterns drive outage predictions.
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
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)

from src.models.model_utils import ModelUtils

# set the style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Create confusion matrix heatmap with both counts and percentages.

    I show both raw counts (for absolute understanding) and
    percentages (for relative performance within each class).
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # plot raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        ax=ax1,
        xticklabels=["No Outage", "Outage"],
        yticklabels=["No Outage", "Outage"],
    )
    ax1.set_title("Confusion Matrix (Raw Counts)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=11)
    ax1.set_xlabel("Predicted", fontsize=11)

    # plot percentages
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        cbar=True,
        square=True,
        ax=ax2,
        xticklabels=["No Outage", "Outage"],
        yticklabels=["No Outage", "Outage"],
    )
    ax2.set_title("Confusion Matrix (% by True Class)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Actual", fontsize=11)
    ax2.set_xlabel("Predicted", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to: {save_path}")


def plot_precision_recall_curve(y_true, y_proba, save_path):
    """
    Plot precision-recall curve with AUC.

    emphasize this curve because it's the most informative metric
    for imbalanced classification (95.64% majority class).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    # find the optimal threshold (F1 score maximization)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point (threshold=1)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    # plot the PR curve
    ax.plot(
        recall,
        precision,
        color="darkblue",
        lw=2,
        label=f"PR Curve (AUC = {pr_auc:.3f})",
    )

    # mark the optimal point
    ax.scatter(
        recall[optimal_idx],
        precision[optimal_idx],
        color="red",
        s=100,
        marker="o",
        label=f"Optimal Threshold = {optimal_threshold:.3f}",
        zorder=5,
    )

    # add baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.plot(
        [0, 1], [baseline, baseline], "k--", lw=1, label=f"Baseline = {baseline:.3f}"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Precision-Recall Curve\n(Primary Metric for Imbalanced Data)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved precision-recall curve to: {save_path}")


def plot_roc_curve(y_true, y_proba, save_path):
    """
    Plot ROC curve with AUC.

    include this as a standard metric, though PR-AUC is more
    informative for our imbalanced dataset.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(10, 6))

    # plot the ROC curve
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})"
    )

    # add the diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {save_path}")


def plot_feature_coefficients(model, feature_names, save_path, top_n=20):
    """
    Plot feature coefficients to show which weather patterns drive predictions.

    I show the top N features by absolute coefficient value, which indicates
    the strength of each feature's influence on outage predictions.
    """
    # extract coefficients
    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_[0]}
    )

    # sort by absolute value to get most influential features
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients_sorted = coefficients.sort_values("abs_coefficient", ascending=False)

    # take top N features
    top_coefficients = coefficients_sorted.head(top_n)

    # create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["green" if c > 0 else "red" for c in top_coefficients["coefficient"]]

    ax.barh(
        range(len(top_coefficients)),
        top_coefficients["coefficient"],
        color=colors,
        alpha=0.7,
    )
    ax.set_yticks(range(len(top_coefficients)))
    ax.set_yticklabels(top_coefficients["feature"], fontsize=9)
    ax.set_xlabel("Coefficient Value", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Coefficients\n(Green = Increases Outage Risk, Red = Decreases)",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved feature coefficients to: {save_path}")

    # also return the top features for reporting
    return top_coefficients


def plot_prediction_distribution(y_true, y_proba, save_path):
    """
    Plot histogram of predicted probabilities by true class.

    I use this to understand how well the model separates the two classes
    and whether there's good calibration.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # separate probabilities by true class
    proba_no_outage = y_proba[y_true == 0]
    proba_outage = y_proba[y_true == 1]

    # plot histograms
    ax.hist(
        proba_no_outage,
        bins=50,
        alpha=0.6,
        color="blue",
        label=f"True No-Outage (n={len(proba_no_outage):,})",
        density=True,
    )
    ax.hist(
        proba_outage,
        bins=50,
        alpha=0.6,
        color="red",
        label=f"True Outage (n={len(proba_outage):,})",
        density=True,
    )

    # add decision threshold line
    ax.axvline(
        x=0.5,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Decision Threshold (0.5)",
    )

    ax.set_xlabel("Predicted Probability of Outage", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Distribution of Predicted Probabilities by True Class",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # add interpretation text
    separation_quality = (
        "good"
        if abs(proba_outage.mean() - proba_no_outage.mean()) > 0.3
        else "moderate"
    )
    ax.text(
        0.5,
        0.95,
        f"Class Separation: {separation_quality.upper()}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved prediction distribution to: {save_path}")


def main():
    """
    Main visualization function.
    """
    print("=" * 80)
    print("MODEL VISUALIZATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # initialize utilities
    utils = ModelUtils()

    # create output directory
    output_dir = utils.results_dir / "figures" / "logistic_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load the trained model
    print("\nLOADING MODEL AND DATA")
    print("-" * 80)
    model, hyperparameters = utils.load_model("logistic_regression_best")

    # load test data
    df, selected_features = utils.load_data_with_features()
    train_df, val_df, test_df = utils.temporal_split(df)
    X_test, y_test = utils.prepare_features_target(test_df, selected_features)

    # generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\nGenerating visualizations for {len(X_test):,} test samples...")

    # generate all visualizations
    print("\nGENERATING VISUALIZATIONS")
    print("-" * 80)

    # 1. Confusion Matrix
    print("\n1. Confusion Matrix...")
    plot_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")

    # 2. Precision-Recall Curve
    print("\n2. Precision-Recall Curve...")
    plot_precision_recall_curve(
        y_test, y_proba, output_dir / "precision_recall_curve.png"
    )

    # 3. ROC Curve
    print("\n3. ROC Curve...")
    plot_roc_curve(y_test, y_proba, output_dir / "roc_curve.png")

    # 4. Feature Coefficients
    print("\n4. Feature Coefficients...")
    top_features = plot_feature_coefficients(
        model, selected_features, output_dir / "feature_coefficients.png", top_n=20
    )

    # 5. Prediction Distribution
    print("\n5. Prediction Distribution...")
    plot_prediction_distribution(
        y_test, y_proba, output_dir / "prediction_distribution.png"
    )

    # save top features summary
    print("\nSAVING FEATURE ANALYSIS")
    print("-" * 80)

    feature_summary = [
        "=" * 80,
        "TOP FEATURES DRIVING OUTAGE PREDICTIONS",
        "=" * 80,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Logistic regression coefficients indicate the strength and direction",
        "of each feature's influence on outage probability:",
        "  - Positive coefficients: Feature increases outage risk",
        "  - Negative coefficients: Feature decreases outage risk",
        "  - Larger absolute value: Stronger influence",
        "",
        "TOP 20 FEATURES BY ABSOLUTE COEFFICIENT VALUE",
        "-" * 80,
        f"{'Rank':<6} {'Feature':<40} {'Coefficient':>12} {'Direction':>12}",
        "-" * 80,
    ]

    for idx, row in top_features.iterrows():
        direction = "Increases" if row["coefficient"] > 0 else "Decreases"
        feature_summary.append(
            f"{idx+1:<6} {row['feature']:<40} {row['coefficient']:>12.4f} {direction:>12}"
        )

    feature_summary.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 80,
            "The features with the largest positive coefficients are the weather",
            "patterns most strongly associated with power outages. These could be:",
            "  - High wind speeds (WSF2, damaging_winds)",
            "  - Extreme temperatures (extreme_heat, extreme_cold)",
            "  - Heavy precipitation (extreme_rain, heavy_rain)",
            "  - Compound risk factors (ice_storm_risk, wet_windy_combo)",
            "",
            "Features with negative coefficients indicate conditions associated",
            "with lower outage risk (e.g., mild weather, stable conditions).",
            "",
        ]
    )

    utils.save_results(
        results="\n".join(feature_summary),
        filename="feature_coefficients_analysis.txt",
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll visualizations saved to:")
    print(f"  {output_dir}")
    print(f"\nGenerated plots:")
    print(f"  1. confusion_matrix.png")
    print(f"  2. precision_recall_curve.png (PRIMARY for imbalanced data)")
    print(f"  3. roc_curve.png")
    print(f"  4. feature_coefficients.png (model interpretability)")
    print(f"  5. prediction_distribution.png")
    print(f"\nFeature analysis saved to:")
    print(f"  results/models/feature_coefficients_analysis.txt")


if __name__ == "__main__":
    main()
