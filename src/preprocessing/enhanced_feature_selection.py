"""
Enhanced Feature Selection for Power Outage Prediction
=======================================================
Uses multiple feature selection methods to identify the most predictive features
for binary classification with class imbalance.

Methods:
- Variance threshold filtering
- Correlation analysis
- Statistical tests (f_classif, mutual_info)
- Tree-based importance (Random Forest)
- Consensus feature selection

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class EnhancedFeatureSelector:
    """
    Comprehensive feature selection using multiple methods.
    Designed for imbalanced binary classification.
    """

    def __init__(self, target_col="any_out", random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.selection_results = {}
        self.feature_scores = {}

    def load_and_validate_data(self, input_path):
        """
        Load the corrected encoded dataset and validate its quality.
        """
        print("=" * 70)
        print("LOADING AND VALIDATING DATA")
        print("=" * 70)

        df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Validation checks
        assert self.target_col in df.columns, f"Target '{self.target_col}' not found"
        assert (
            df[self.target_col].nunique() == 2
        ), f"Need both classes, found {df[self.target_col].nunique()}"

        print(f"\nTarget distribution:")
        print(df[self.target_col].value_counts())
        print(f"Outage rate: {df[self.target_col].mean():.2%}")

        if "fips_code" in df.columns:
            print(f"Counties: {df['fips_code'].nunique()}")

        print("\nValidation: PASSED")

        return df

    def prepare_features(self, df):
        """
        Separate features from target and exclude leakage columns.
        """
        print("\n" + "=" * 70)
        print("FEATURE PREPARATION")
        print("=" * 70)

        # Define columns to exclude (target and leakage)
        exclude_features = [
            self.target_col,
            "num_out_per_day",
            "minutes_out",
            "customers_out",
            "customers_out_mean",
            "cust_minute_area",
            "pct_out_max",
            "pct_out_area",
            "pct_out_area_unified",
            "pct_out_area_covered",
            "pct_out_max_unified",
            "train_mask",
        ]

        # Keep only features that exist in df
        exclude_features = [f for f in exclude_features if f in df.columns]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_features]

        print(f"Total columns: {len(df.columns)}")
        print(f"Excluded columns: {len(exclude_features)}")
        print(f"  - {', '.join(exclude_features[:5])}...")
        print(f"Feature columns: {len(feature_cols)}")

        X = df[feature_cols]
        y = df[self.target_col]

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print(f"\nHandling {X.isnull().sum().sum()} missing values...")
            X = X.fillna(X.mean())

        return X, y, feature_cols

    def variance_filter(self, X, threshold=0.01):
        """
        Remove features with low variance.
        """
        print("\n" + "=" * 70)
        print("METHOD 1: VARIANCE THRESHOLD")
        print("=" * 70)

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)

        low_variance_mask = selector.get_support()
        selected_features = X.columns[low_variance_mask].tolist()
        removed_features = X.columns[~low_variance_mask].tolist()

        print(f"Threshold: {threshold}")
        print(f"Features removed: {len(removed_features)}")
        print(f"Features remaining: {len(selected_features)}")

        self.selection_results["variance"] = selected_features

        return selected_features, X[selected_features]

    def correlation_filter(self, X, threshold=0.95):
        """
        Remove highly correlated features to reduce multicollinearity.
        """
        print("\n" + "=" * 70)
        print("METHOD 2: CORRELATION FILTER")
        print("=" * 70)

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Find highly correlated pairs
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identify features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        selected_features = [col for col in X.columns if col not in to_drop]

        print(f"Threshold: {threshold}")
        print(f"Features removed: {len(to_drop)}")
        print(f"Features remaining: {len(selected_features)}")

        self.selection_results["correlation"] = selected_features

        return selected_features, X[selected_features]

    def statistical_selection(self, X, y, k=50):
        """
        Statistical feature selection using ANOVA F-value and mutual information.
        """
        print("\n" + "=" * 70)
        print("METHOD 3: STATISTICAL TESTS")
        print("=" * 70)

        # SelectKBest with f_classif (ANOVA F-value)
        print("\n3a. ANOVA F-value (f_classif):")
        selector_f = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        selector_f.fit(X, y)

        features_f = X.columns[selector_f.get_support()].tolist()
        scores_f = selector_f.scores_

        print(f"  Selected: {len(features_f)} features")

        # SelectKBest with mutual_info_classif
        print("\n3b. Mutual Information:")
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        selector_mi.fit(X, y)

        features_mi = X.columns[selector_mi.get_support()].tolist()
        scores_mi = selector_mi.scores_

        print(f"  Selected: {len(features_mi)} features")

        # Store scores
        self.feature_scores["f_classif"] = pd.DataFrame(
            {"feature": X.columns, "score": scores_f}
        ).sort_values("score", ascending=False)

        self.feature_scores["mutual_info"] = pd.DataFrame(
            {"feature": X.columns, "score": scores_mi}
        ).sort_values("score", ascending=False)

        # Combine both methods
        combined_features = list(set(features_f + features_mi))
        print(f"\nCombined (union): {len(combined_features)} features")

        self.selection_results["f_classif"] = features_f
        self.selection_results["mutual_info"] = features_mi
        self.selection_results["statistical_combined"] = combined_features

        return combined_features

    def tree_based_selection(self, X, y, k=50):
        """
        Tree-based feature importance using Random Forest with balanced classes.
        """
        print("\n" + "=" * 70)
        print("METHOD 4: TREE-BASED IMPORTANCE")
        print("=" * 70)

        # Random Forest with class balancing
        print("\n4a. Random Forest (balanced classes):")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )

        rf.fit(X, y)

        # Get feature importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Select top k features
        selected_features = feature_importance.head(k)["feature"].tolist()

        print(f"  Selected: {len(selected_features)} features")
        print(f"  Top 5 features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    - {row['feature']}: {row['importance']:.4f}")

        self.feature_scores["random_forest"] = feature_importance
        self.selection_results["random_forest"] = selected_features

        return selected_features

    def get_consensus_features(self, min_methods=2):
        """
        Get features selected by at least min_methods.
        """
        print("\n" + "=" * 70)
        print("CONSENSUS FEATURE SELECTION")
        print("=" * 70)

        # Count how many methods selected each feature
        all_features = set()
        for features in self.selection_results.values():
            all_features.update(features)

        feature_counts = {}
        for feature in all_features:
            count = sum(
                1 for features in self.selection_results.values() if feature in features
            )
            feature_counts[feature] = count

        # Select features appearing in at least min_methods
        consensus_features = [
            f for f, count in feature_counts.items() if count >= min_methods
        ]

        print(
            f"Features selected by at least {min_methods} methods: {len(consensus_features)}"
        )
        print(f"\nMethod coverage:")
        for method, features in self.selection_results.items():
            overlap = len([f for f in consensus_features if f in features])
            print(f"  {method:25s}: {overlap:3d}/{len(features):3d} features")

        # Sort by frequency
        consensus_with_counts = [(f, feature_counts[f]) for f in consensus_features]
        consensus_with_counts.sort(key=lambda x: x[1], reverse=True)

        return consensus_features, feature_counts

    def categorize_features(self, features):
        """
        Categorize selected features into types.
        """
        categories = {
            "weather_base": [],
            "weather_types": [],
            "temporal": [],
            "extreme_indicators": [],
            "compound_risks": [],
            "geographic": [],
            "other": [],
        }

        for feature in features:
            if any(
                x in feature for x in ["AWND", "PRCP", "TMAX", "TMIN", "WSF2", "WSF5"]
            ):
                categories["weather_base"].append(feature)
            elif feature.startswith("WT"):
                categories["weather_types"].append(feature)
            elif any(
                x in feature
                for x in [
                    "lag",
                    "_3d_",
                    "_7d_",
                    "_14d_",
                    "day_of_",
                    "month",
                    "year",
                    "season",
                ]
            ):
                categories["temporal"].append(feature)
            elif any(
                x in feature
                for x in [
                    "extreme",
                    "heavy",
                    "damaging",
                    "high_winds",
                    "freezing",
                    "heat_wave",
                ]
            ):
                categories["extreme_indicators"].append(feature)
            elif any(
                x in feature
                for x in [
                    "ice_storm",
                    "wet_windy",
                    "multiple_extremes",
                    "thermal_stress",
                    "mechanical_stress",
                ]
            ):
                categories["compound_risks"].append(feature)
            elif any(
                x in feature
                for x in ["county", "fips", "state", "climate", "risk_profile"]
            ):
                categories["geographic"].append(feature)
            else:
                categories["other"].append(feature)

        return categories

    def visualize_importance(self, output_dir):
        """
        Generate feature importance visualizations.
        """
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Random Forest Feature Importance (Top 30)
        if "random_forest" in self.feature_scores:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_scores["random_forest"].head(30)
            plt.barh(range(len(top_features)), top_features["importance"])
            plt.yticks(range(len(top_features)), top_features["feature"])
            plt.xlabel("Feature Importance")
            plt.title("Top 30 Features by Random Forest Importance")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                output_dir / "feature_importance_rf.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            print(f"Saved: {output_dir / 'feature_importance_rf.png'}")

        # Plot 2: Comparison across methods
        plt.figure(figsize=(12, 6))
        methods = list(self.selection_results.keys())
        counts = [len(features) for features in self.selection_results.values()]

        plt.bar(range(len(methods)), counts)
        plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
        plt.ylabel("Number of Features Selected")
        plt.title("Feature Count by Selection Method")
        plt.tight_layout()
        plt.savefig(output_dir / "method_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / 'method_comparison.png'}")

    def export_results(
        self, output_dir, consensus_features, feature_counts, categories
    ):
        """
        Export selected features and comprehensive report.
        """
        print("\n" + "=" * 70)
        print("EXPORTING RESULTS")
        print("=" * 70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export 1: Selected features CSV
        consensus_df = pd.DataFrame(
            [
                {"feature": f, "selected_by_n_methods": feature_counts[f]}
                for f in consensus_features
            ]
        ).sort_values("selected_by_n_methods", ascending=False)

        csv_path = output_dir / "selected_features.csv"
        consensus_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # Export 2: Feature scores from all methods
        scores_path = output_dir / "feature_scores_all_methods.csv"
        all_scores = pd.DataFrame(
            {"feature": list(self.feature_scores["f_classif"]["feature"])}
        )

        for method, scores_df in self.feature_scores.items():
            all_scores = all_scores.merge(
                scores_df.rename(columns={"score": f"{method}_score"}),
                on="feature",
                how="left",
            )

        all_scores.to_csv(scores_path, index=False)
        print(f"Saved: {scores_path}")

        # Export 3: Comprehensive report
        report_path = output_dir / "feature_selection_report.md"
        with open(report_path, "w") as f:
            f.write("# Enhanced Feature Selection Report\n\n")
            f.write("**Generated**: 2025-10-26\n\n")
            f.write("---\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total features in dataset: {len(feature_counts)}\n")
            f.write(f"- Consensus features selected: {len(consensus_features)}\n")
            f.write(f"- Minimum methods required: 2\n\n")

            f.write("## Selection Methods Used\n\n")
            for method, features in self.selection_results.items():
                f.write(f"- **{method}**: {len(features)} features\n")

            f.write("\n## Feature Categories\n\n")
            for category, features in categories.items():
                if features:
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    f.write(f"Count: {len(features)}\n\n")
                    for feat in sorted(features):
                        count = feature_counts.get(feat, 0)
                        f.write(f"- {feat} (selected by {count} methods)\n")
                    f.write("\n")

            f.write("## Top 20 Features by Selection Frequency\n\n")
            top_20 = sorted(
                [(f, feature_counts[f]) for f in consensus_features],
                key=lambda x: x[1],
                reverse=True,
            )[:20]

            f.write("| Rank | Feature | Selected by N Methods |\n")
            f.write("|------|---------|----------------------|\n")
            for idx, (feat, count) in enumerate(top_20, 1):
                f.write(f"| {idx} | {feat} | {count} |\n")

        print(f"Saved: {report_path}")


def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 70)
    print("ENHANCED FEATURE SELECTION")
    print("=" * 70)

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    input_path = (
        project_root
        / "data"
        / "ml_ready"
        / "merged_weather_outages_2019_2024_encoded.csv"
    )
    output_dir = project_root / "results" / "feature_selection"

    # Initialize selector
    selector = EnhancedFeatureSelector(target_col="any_out", random_state=42)

    # Load and validate data
    df = selector.load_and_validate_data(input_path)

    # Prepare features
    X, y, feature_cols = selector.prepare_features(df)

    # Apply selection methods
    var_features, X_var = selector.variance_filter(X, threshold=0.01)
    corr_features, X_corr = selector.correlation_filter(X_var, threshold=0.95)
    stat_features = selector.statistical_selection(X_corr, y, k=50)
    tree_features = selector.tree_based_selection(X_corr, y, k=50)

    # Get consensus features
    consensus_features, feature_counts = selector.get_consensus_features(min_methods=2)

    # Categorize features
    categories = selector.categorize_features(consensus_features)

    print("\n" + "=" * 70)
    print("FEATURE CATEGORIES")
    print("=" * 70)
    for category, features in categories.items():
        if features:
            print(
                f"{category.replace('_', ' ').title():25s}: {len(features):3d} features"
            )

    # Generate visualizations
    selector.visualize_importance(output_dir)

    # Export results
    selector.export_results(output_dir, consensus_features, feature_counts, categories)

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 70)
    print(f"\nSelected {len(consensus_features)} features for modeling")
    print(f"Results saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review selected features in: results/feature_selection/")
    print("  2. Use selected_features.csv for model training")
    print("  3. Check feature_selection_report.md for detailed analysis")


if __name__ == "__main__":
    main()
