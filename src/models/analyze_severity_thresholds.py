"""
Severity Threshold Analysis for Multiclass Classification

According to Manav's idea, I analyze different threshold values to find the optimal split for creating
3-class severity labels (No Outage, Minor, Severe) with the most balanced
class distribution.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_utils import ModelUtils


def create_severity_classes(df, threshold, col="pct_out_area_unified"):
    """
    3-class severity target variable based on threshold.

    Class 0: No outage (pct_out_area_unified == 0)
    Class 1: Minor outage (0 < pct_out_area_unified < threshold)
    Class 2: Severe outage (pct_out_area_unified >= threshold)
    """
    conditions = [
        df[col] == 0,
        (df[col] > 0) & (df[col] < threshold),
        df[col] >= threshold,
    ]
    choices = [0, 1, 2]
    return np.select(conditions, choices, default=0)


def calculate_balance_metrics(class_counts):
    """
    Calculate balance metrics for class distribution.

    Returns:
        min_pct: Minimum class percentage (higher is better)
        std_pct: Standard deviation of class percentages (lower is better)
        entropy: Entropy-based balance score (higher is better)
        balance_score: Combined score (higher is better)
    """
    percentages = class_counts / class_counts.sum()
    min_pct = percentages.min()
    std_pct = percentages.std()
    entropy = -np.sum(percentages * np.log(percentages + 1e-10))
    max_entropy = np.log(len(percentages))
    normalized_entropy = entropy / max_entropy
    balance_score = min_pct * (1 - std_pct) * normalized_entropy

    return {
        "min_pct": min_pct,
        "std_pct": std_pct,
        "entropy": normalized_entropy,
        "balance_score": balance_score,
    }


def analyze_thresholds(df, thresholds, col="pct_out_area_unified"):
    """
    Analyze all thresholds and return results.
    """
    results = []

    print("Analyzing thresholds...")
    print("-" * 80)

    for threshold in thresholds:
        severity_classes = create_severity_classes(df, threshold, col)
        class_counts = pd.Series(severity_classes).value_counts().sort_index()

        class_dist = {
            0: class_counts.get(0, 0),
            1: class_counts.get(1, 0),
            2: class_counts.get(2, 0),
        }

        total = sum(class_dist.values())
        class_pcts = {k: v / total for k, v in class_dist.items()}
        metrics = calculate_balance_metrics(pd.Series(class_dist))

        result = {
            "threshold": threshold,
            "threshold_pct": f"{threshold:.0%}",
            "class_0_count": class_dist[0],
            "class_1_count": class_dist[1],
            "class_2_count": class_dist[2],
            "class_0_pct": class_pcts[0],
            "class_1_pct": class_pcts[1],
            "class_2_pct": class_pcts[2],
            "min_class_pct": metrics["min_pct"],
            "std_class_pct": metrics["std_pct"],
            "entropy": metrics["entropy"],
            "balance_score": metrics["balance_score"],
        }

        results.append(result)

        print(f"\nThreshold: {threshold:.0%}")
        print(f"  Class 0 (No Outage):    {class_dist[0]:6,} ({class_pcts[0]:6.2%})")
        print(f"  Class 1 (Minor):        {class_dist[1]:6,} ({class_pcts[1]:6.2%})")
        print(f"  Class 2 (Severe):       {class_dist[2]:6,} ({class_pcts[2]:6.2%})")
        print(f"  Min class %:            {metrics['min_pct']:.4f}")
        print(f"  Std dev:                {metrics['std_pct']:.4f}")
        print(f"  Balance score:          {metrics['balance_score']:.4f}")

    return pd.DataFrame(results)


def find_best_threshold(results_df):
    """
    Find the threshold with the most balanced classes.

    Prioritizes highest minimum class percentage and lowest std deviation.
    """
    results_df_sorted = results_df.sort_values(
        by=["balance_score", "min_class_pct"], ascending=[False, False]
    )

    best = results_df_sorted.iloc[0]
    return best["threshold"]


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("SEVERITY THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    utils = ModelUtils()

    print("Loading data...")
    print("-" * 80)
    data_path = utils.data_dir / "merged_weather_outages_2019_2024_encoded.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    if "pct_out_area_unified" not in df.columns:
        raise ValueError("Column 'pct_out_area_unified' not found in dataset")

    print(f"\nOutage percentage statistics:")
    print(f"  Min:  {df['pct_out_area_unified'].min():.6f}")
    print(f"  Max:  {df['pct_out_area_unified'].max():.6f}")
    print(f"  Mean: {df['pct_out_area_unified'].mean():.6f}")
    print(f"  Median: {df['pct_out_area_unified'].median():.6f}")

    no_outage_count = (df["pct_out_area_unified"] == 0).sum()
    print(f"\nDays with no outage: {no_outage_count:,} ({no_outage_count/len(df):.2%})")

    print("\nAnalyzing thresholds...")
    print("-" * 80)

    thresholds = [0.01, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]
    results_df = analyze_thresholds(df, thresholds)

    print("\nSelecting best threshold...")
    print("-" * 80)

    best_threshold = find_best_threshold(results_df)
    best_row = results_df[results_df["threshold"] == best_threshold].iloc[0]

    print(f"\nBest threshold: {best_threshold:.0%}")
    print(f"Class distribution:")
    print(
        f"  No Outage (0):  {best_row['class_0_count']:,} ({best_row['class_0_pct']:.2%})"
    )
    print(
        f"  Minor (1):      {best_row['class_1_count']:,} ({best_row['class_1_pct']:.2%})"
    )
    print(
        f"  Severe (2):     {best_row['class_2_count']:,} ({best_row['class_2_pct']:.2%})"
    )
    print(f"\nBalance metrics:")
    print(f"  Min class %:    {best_row['min_class_pct']:.4f}")
    print(f"  Std dev:        {best_row['std_class_pct']:.4f}")
    print(f"  Balance score:  {best_row['balance_score']:.4f}")

    print("\nSaving results...")
    print("-" * 80)

    utils.results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = utils.results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = models_dir / "threshold_analysis.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved threshold analysis to: {csv_path}")

    json_path = models_dir / "best_threshold.json"
    threshold_data = {
        "best_threshold": float(best_threshold),
        "best_threshold_pct": f"{best_threshold:.0%}",
        "selected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "class_distribution": {
            "no_outage_count": int(best_row["class_0_count"]),
            "minor_count": int(best_row["class_1_count"]),
            "severe_count": int(best_row["class_2_count"]),
            "no_outage_pct": float(best_row["class_0_pct"]),
            "minor_pct": float(best_row["class_1_pct"]),
            "severe_pct": float(best_row["class_2_pct"]),
        },
        "balance_metrics": {
            "min_class_pct": float(best_row["min_class_pct"]),
            "std_class_pct": float(best_row["std_class_pct"]),
            "balance_score": float(best_row["balance_score"]),
        },
    }

    with open(json_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Saved best threshold to: {json_path}")




if __name__ == "__main__":
    main()
