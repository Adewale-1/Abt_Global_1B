"""
Enhanced Feature Selection for Power Outage *Regression*
========================================================
Target: pct_out_area_unified (continuous)

Methods:
- Variance threshold filtering
- Correlation analysis
- Statistical tests (f_regression, mutual_info_regression)
- Tree-based importance (RandomForestRegressor)
- Consensus feature selection (min_methods=2)

Artifacts (saved under results/feature_selection/):
- selected_features_regression.csv
- feature_scores_all_methods_regression.csv
- feature_selection_report_regression.md
- feature_importance_rf_regression.png
- method_comparison_regression.png
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    mutual_info_regression,
)
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


class EnhancedFeatureSelectorRegression:
    def __init__(self, target_col: str = "pct_out_area_unified", random_state: int = 42):
        self.target_col = target_col
        self.random_state = random_state
        self.selection_results = {}
        self.feature_scores = {}

    # -----------------------------
    # Data load & validation
    # -----------------------------
    def load_and_validate_data(self, input_path: Path) -> pd.DataFrame:
        print("=" * 70)
        print("LOADING AND VALIDATING DATA (REGRESSION)")
        print("=" * 70)

        df = pd.read_csv(input_path, low_memory=False)
        print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

        assert self.target_col in df.columns, f"Target '{self.target_col}' not found"
        if "fips_code" in df.columns:
            print(f"Counties: {df['fips_code'].nunique()}")
        print(f"{self.target_col} summary:")
        print(df[self.target_col].describe())

        print("\nValidation: PASSED")
        return df

    # -----------------------------
    # Feature prep (exclude leakage)
    # -----------------------------
    def prepare_features(self, df: pd.DataFrame):
        print("\n" + "=" * 70)
        print("FEATURE PREPARATION (REGRESSION)")
        print("=" * 70)

        # Exclude target + leakage label variants (classification or other targets)
        exclude_features = [
            self.target_col,
            # Outage label family (binary + related engineered summaries)
            "any_out",
            "num_out_per_day",
            "minutes_out",
            "customers_out",
            "customers_out_mean",
            "cust_minute_area",
            # Other pct/rate cousins (avoid leakage)
            "pct_out_max",
            "pct_out_area",
            "pct_out_area_unified",
            "pct_out_area_covered",
            "pct_out_max_unified",
            "train_mask",
        ]
        exclude_features = [c for c in exclude_features if c in df.columns]

        feature_cols = [c for c in df.columns if c not in exclude_features]
        print(f"Total columns: {len(df.columns)}")
        print(f"Excluded columns: {len(exclude_features)}")
        if exclude_features:
            print("  - " + ", ".join(exclude_features[:8]) + ("..." if len(exclude_features) > 8 else ""))
        print(f"Feature columns: {len(feature_cols)}")

        X = df[feature_cols]
        y = df[self.target_col]

        # Handle missing
        if X.isnull().sum().sum() > 0:
            print(f"\nHandling {X.isnull().sum().sum()} missing values...")
            X = X.fillna(X.mean())

        return X, y, feature_cols

    # -----------------------------
    # Method 1: Variance threshold
    # -----------------------------
    def variance_filter(self, X: pd.DataFrame, threshold: float = 0.01):
        print("\n" + "=" * 70)
        print("METHOD 1: VARIANCE THRESHOLD")
        print("=" * 70)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        mask = selector.get_support()
        kept = X.columns[mask].tolist()
        removed = X.columns[~mask].tolist()
        print(f"Threshold: {threshold}")
        print(f"Features removed: {len(removed)}")
        print(f"Features remaining: {len(kept)}")
        self.selection_results["variance"] = kept
        return kept, X[kept]

    # -----------------------------
    # Method 2: Correlation filter
    # -----------------------------
    def correlation_filter(self, X: pd.DataFrame, threshold: float = 0.95):
        print("\n" + "=" * 70)
        print("METHOD 2: CORRELATION FILTER")
        print("=" * 70)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
        kept = [c for c in X.columns if c not in to_drop]
        print(f"Threshold: {threshold}")
        print(f"Features removed: {len(to_drop)}")
        print(f"Features remaining: {len(kept)}")
        self.selection_results["correlation"] = kept
        return kept, X[kept]

    # -----------------------------
    # Method 3: Statistical tests
    # -----------------------------
    def statistical_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50):
        print("\n" + "=" * 70)
        print("METHOD 3: STATISTICAL TESTS (REGRESSION)")
        print("=" * 70)

        print("\n3a. f_regression (ANOVA analog for regression):")
        sel_f = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        sel_f.fit(X, y)
        feats_f = X.columns[sel_f.get_support()].tolist()
        scores_f = sel_f.scores_
        print(f"  Selected: {len(feats_f)}")

        print("\n3b. mutual_info_regression:")
        sel_mi = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
        sel_mi.fit(X, y)
        feats_mi = X.columns[sel_mi.get_support()].tolist()
        scores_mi = sel_mi.scores_
        print(f"  Selected: {len(feats_mi)}")

        self.feature_scores["f_regression"] = (
            pd.DataFrame({"feature": X.columns, "score": scores_f})
            .sort_values("score", ascending=False)
        )
        self.feature_scores["mutual_info_regression"] = (
            pd.DataFrame({"feature": X.columns, "score": scores_mi})
            .sort_values("score", ascending=False)
        )

        combined = list(set(feats_f + feats_mi))
        print(f"\nCombined (union): {len(combined)}")
        self.selection_results["f_regression"] = feats_f
        self.selection_results["mutual_info_regression"] = feats_mi
        self.selection_results["statistical_combined"] = combined
        return combined

    # -----------------------------
    # Method 4: Tree-based (Regressor)
    # -----------------------------
    def tree_based_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50):
        print("\n" + "=" * 70)
        print("METHOD 4: TREE-BASED IMPORTANCE (RandomForestRegressor)")
        print("=" * 70)

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=20,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}) \
            .sort_values("importance", ascending=False)
        selected = fi.head(k)["feature"].tolist()
        print(f"  Selected: {len(selected)}")
        print("  Top 5:")
        for _, row in fi.head(5).iterrows():
            print(f"    - {row['feature']}: {row['importance']:.4f}")

        self.feature_scores["random_forest_regressor"] = fi
        self.selection_results["random_forest_regressor"] = selected
        return selected

    # -----------------------------
    # Consensus
    # -----------------------------
    def get_consensus_features(self, min_methods: int = 2):
        print("\n" + "=" * 70)
        print("CONSENSUS FEATURE SELECTION (REGRESSION)")
        print("=" * 70)

        all_feats = set()
        for feats in self.selection_results.values():
            all_feats.update(feats)

        counts = {f: sum(f in feats for feats in self.selection_results.values())
                  for f in all_feats}
        consensus = [f for f, c in counts.items() if c >= min_methods]

        print(f"Features selected by ≥{min_methods} methods: {len(consensus)}")
        print("\nMethod coverage:")
        for method, feats in self.selection_results.items():
            overlap = sum(1 for f in consensus if f in feats)
            print(f"  {method:28s}: {overlap:3d}/{len(feats):3d}")
        return consensus, counts

    # -----------------------------
    # Visuals
    # -----------------------------
    def visualize_importance(self, output_dir: Path):
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS (REGRESSION)")
        print("=" * 70)
        output_dir.mkdir(parents=True, exist_ok=True)

        # RF importances
        if "random_forest_regressor" in self.feature_scores:
            plt.figure(figsize=(12, 8))
            top = self.feature_scores["random_forest_regressor"].head(30)
            plt.barh(range(len(top)), top["importance"])
            plt.yticks(range(len(top)), top["feature"])
            plt.gca().invert_yaxis()
            plt.xlabel("Importance")
            plt.title("Top 30 Features (RandomForestRegressor)")
            plt.tight_layout()
            path = output_dir / "feature_importance_rf_regression.png"
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {path}")

        # Method comparison
        plt.figure(figsize=(12, 6))
        methods = list(self.selection_results.keys())
        counts = [len(v) for v in self.selection_results.values()]
        plt.bar(range(len(methods)), counts)
        plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
        plt.ylabel("Number of Features Selected")
        plt.title("Feature Count by Selection Method (Regression)")
        plt.tight_layout()
        path = output_dir / "method_comparison_regression.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

     # -----------------------------
    # Export
    # -----------------------------
    def export_results(self, output_dir, consensus_features, feature_counts, categories=None):
        """
        Export selected features and a comprehensive report for regression.
        """
        print("\n" + "=" * 70)
        print("EXPORTING RESULTS (REGRESSION)")
        print("=" * 70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Selected features CSV
        consensus_df = (
            pd.DataFrame(
                [{"feature": f, "selected_by_n_methods": feature_counts[f]} for f in consensus_features]
            )
            .sort_values("selected_by_n_methods", ascending=False)
            .reset_index(drop=True)
        )
        csv_path = output_dir / "selected_features_regression.csv"
        consensus_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        # 2) All method scores into one table
        # Build a master list of features from all scoreframes
        all_feats = set()
        for df in self.feature_scores.values():
            all_feats.update(df["feature"].tolist())
        all_scores = pd.DataFrame({"feature": sorted(all_feats)})

        # Merge each method with a method-specific column name
        for method, df in self.feature_scores.items():
            df = df.copy()
            if "score" in df.columns:
                new_col = f"{method}_score"
                df = df.rename(columns={"score": new_col})[["feature", new_col]]
            elif "importance" in df.columns:
                new_col = f"{method}_importance"
                df = df.rename(columns={"importance": new_col})[["feature", new_col]]
            else:
                print(f"Warning: method '{method}' has no 'score' or 'importance' column; skipping.")
                continue
            all_scores = all_scores.merge(df, on="feature", how="left")

        scores_path = output_dir / "feature_scores_all_methods_regression.csv"
        all_scores.to_csv(scores_path, index=False)
        print(f"Saved: {scores_path}")

        # 3) Markdown report
        report_path = output_dir / "feature_selection_report_regression.md"
        with open(report_path, "w") as f:
            f.write("Enhanced Feature Selection Report (Regression)\n")
            f.write("Generated: 2025-10-30\n\n")
            f.write("Summary\n")
            f.write(f"Total features in dataset: {len(feature_counts)}\n")
            f.write(f"Consensus features selected: {len(consensus_features)}\n")
            f.write("Minimum methods required: 2\n\n")

            f.write("Selection Methods Used\n")
            for method, feats in self.selection_results.items():
                f.write(f"{method}: {len(feats)} features\n")
            f.write("\n")

            if categories:
                f.write("Feature Categories\n")
                for cat, feats in categories.items():
                    if feats:
                        f.write(f"{cat.replace('_',' ').title()}\n")
                        f.write(f"Count: {len(feats)}\n\n")
                        for feat in sorted(feats):
                            count = feature_counts.get(feat, 0)
                            f.write(f"{feat} (selected by {count} methods)\n")
                        f.write("\n")

            f.write("Top 20 Features by Selection Frequency\n")
            top_20 = sorted(
                [(ftr, feature_counts[ftr]) for ftr in consensus_features],
                key=lambda x: x[1],
                reverse=True,
            )[:20]
            f.write("Rank\tFeature\tSelected by N Methods\n")
            for idx, (feat, cnt) in enumerate(top_20, 1):
                f.write(f"{idx}\t{feat}\t{cnt}\n")

        print(f"Saved: {report_path}")

def main():
    print("\n" + "=" * 70)
    print("ENHANCED FEATURE SELECTION (REGRESSION)")
    print("=" * 70)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    input_path = project_root / "data" / "ml_ready" / "merged_weather_outages_2019_2024_encoded.csv"
    output_dir = project_root / "results" / "feature_selection"

    selector = EnhancedFeatureSelectorRegression(target_col="pct_out_area_unified", random_state=42)
    df = selector.load_and_validate_data(input_path)
    X, y, _ = selector.prepare_features(df)

    var_feats, Xv = selector.variance_filter(X, threshold=0.01)
    corr_feats, Xc = selector.correlation_filter(Xv, threshold=0.95)
    stat_feats = selector.statistical_selection(Xc, y, k=50)
    tree_feats = selector.tree_based_selection(Xc, y, k=50)

    consensus, counts = selector.get_consensus_features(min_methods=2)
    selector.visualize_importance(output_dir)
    selector.export_results(output_dir, consensus, counts)

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE (REGRESSION)")
    print("=" * 70)
    print(f"Selected {len(consensus)} features. Results in: {output_dir}")


if __name__ == "__main__":
    main()
