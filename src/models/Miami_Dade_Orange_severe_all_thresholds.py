import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from src.models.model_utils import ModelUtils

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def find_balanced_threshold(pct_series, thresholds):
    """Find threshold that creates most balanced severe/non-severe split."""
    results = []
    for t in thresholds:
        severe = (pct_series >= t).sum()
        non_severe = (pct_series < t).sum()
        total = len(pct_series)
        ratio = severe / total if total > 0 else 0
        balance_score = abs(0.5 - ratio)
        results.append(
            {
                "threshold": float(t),
                "non_severe": int(non_severe),
                "severe": int(severe),
                "severe_ratio": float(ratio),
                "balance_score": float(balance_score),
            }
        )
    return pd.DataFrame(results)


def main():
    print("MIAMI-DADE (FL) & ORANGE (FL) SEVERE OUTAGE CLASSIFICATION - ALL THRESHOLDS")

    utils = ModelUtils()

    print("Loading encoded dataset...")
    df, selected_features = utils.load_data_with_features()
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    print("\nLoading selected features...")
    print(f"Using {len(selected_features)} selected features")

    pct_col = "pct_out_area_unified"

    # Target counties
    target_counties = [
        {"fips": 12086, "name": "Miami_Dade_County_FL"},
        {"fips": 12095, "name": "Orange_County_FL"},
    ]

    # Compute county medians for thresholds
    print("\n--- COUNTY MEDIANS ---")
    md_median = df[df["fips_code"] == 12086][pct_col].median()
    orange_median = df[df["fips_code"] == 12095][pct_col].median()
    print(f"Miami_Dade median pct_out: {md_median:.6f}")
    print(f"Orange_FL median pct_out: {orange_median:.6f}")

    # Thresholds to test (percentage of customers affected)
    # Includes median-based thresholds + industry-ish standards
    test_thresholds = [
        0.0001,      # 0.01%
        md_median,   # Miami-Dade median
        orange_median,  # Orange median
        0.0003,      # 0.03%
        0.0005,      # 0.05%
        0.001,       # 0.1%
        0.005,       # 0.5%
        0.01,        # 1%
    ]

    # Where to save stuff
    results_dir = utils.results_dir / "county_models"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_dir = utils.models_dir / "county_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameter search space
    param_distributions = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000, 2000],
        "class_weight": ["balanced"],
    }

    all_results = {}

    for county in target_counties:
        fips = county["fips"]
        name = county["name"]

        print("\n\n==============================")
        print(f"Processing {name} ({fips})")
        print("==============================")

        county_df = df[df["fips_code"] == fips].copy()
        if len(county_df) == 0:
            print(f"No data found for {name}! Skipping.")
            continue

        print(f"Total samples: {len(county_df):,}")

        # Threshold analysis for overall balance
        print("\n--- THRESHOLD ANALYSIS (overall balance) ---")
        threshold_results = find_balanced_threshold(county_df[pct_col], test_thresholds)
        print(
            f"\n{'Threshold':>12} | {'Non-Severe':>12} | {'Severe':>12} | {'Severe %':>10}"
        )
        print("-" * 55)
        for _, row in threshold_results.iterrows():
            print(
                f"{row['threshold']:>12.4f} | {row['non_severe']:>12,} | {row['severe']:>12,} | {row['severe_ratio']:>10.2%}"
            )

        best_idx = threshold_results["balance_score"].idxmin()
        best_balance_threshold = float(threshold_results.loc[best_idx, "threshold"])
        print(
            f"\nMost balanced threshold by class counts: "
            f"{best_balance_threshold:.4f} ({best_balance_threshold*100:.2f}%)"
        )
        print(f"  Severe ratio: {threshold_results.loc[best_idx, 'severe_ratio']:.2%}")

        county_threshold_models = []
        best_f1 = -1.0
        best_f1_threshold = None

        # Train a model for EACH candidate threshold
        for t in test_thresholds:
            t = float(t)
            print(
                f"\n>>> Training model for {name} with threshold {t:.6f} ({t*100:.3f}%)"
            )

            df_t = county_df.copy()
            df_t["severe_outage"] = (df_t[pct_col] >= t).astype(int)
            target_col = "severe_outage"

            # Time-based split: Test = 2022, Train/Val = other years
            test_mask = df_t["year"] == 2022
            train_val_mask = ~test_mask

            test_df = df_t[test_mask].copy()
            train_val_df = df_t[train_val_mask].copy()

            # Need both classes in train
            if train_val_df[target_col].nunique() < 2:
                print(
                    f"  Skipping threshold {t:.6f}: training data has only one class."
                )
                continue

            if test_df[target_col].nunique() < 2:
                print(
                    f"  Warning: test set for threshold {t:.6f} has only one class. "
                    f"Some metrics (ROC/PR) may be degenerate."
                )

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=0.2,
                random_state=42,
                stratify=train_val_df[target_col],
            )

            print(f"  Train set: {len(train_df):,} samples")
            print(f"  Val set:   {len(val_df):,} samples")
            print(f"  Test set:  {len(test_df):,} samples (Year 2022)")

            X_train = train_df[selected_features]
            y_train = train_df[target_col]
            X_val = val_df[selected_features]
            y_val = val_df[target_col]
            X_test = test_df[selected_features]
            y_test = test_df[target_col]

            print("  Class Distribution (Train):")
            print(
                f"    Non-Severe (0): {(y_train == 0).sum():,} ({(y_train == 0).mean():.2%})"
            )
            print(
                f"    Severe (1):     {(y_train == 1).sum():,} ({(y_train == 1).mean():.2%})"
            )
            print("  Class Distribution (Test):")
            print(
                f"    Non-Severe (0): {(y_test == 0).sum():,} ({(y_test == 0).mean():.2%})"
            )
            print(
                f"    Severe (1):     {(y_test == 1).sum():,} ({(y_test == 1).mean():.2%})"
            )

            if y_train.nunique() < 2:
                print(
                    f"  Skipping threshold {t:.6f}: train set has only one class after split."
                )
                continue

            # Hyperparameter tuning
            print("  --- HYPERPARAMETER TUNING ---")
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            search = RandomizedSearchCV(
                estimator=LogisticRegression(random_state=42),
                param_distributions=param_distributions,
                n_iter=20,
                cv=cv_strategy,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
                random_state=42,
                return_train_score=True,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
            print("  Best Hyperparameters:")
            for k, v in search.best_params_.items():
                print(f"    {k}: {v}")

            # Evaluation on 2022 test set
            print("\n  --- EVALUATION (2022 Test Set) ---")
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            if y_test.nunique() > 1:
                roc_auc = roc_auc_score(y_test, y_proba)
                pr_auc = average_precision_score(y_test, y_proba)
            else:
                roc_auc = 0.0
                pr_auc = 0.0
                print("  Warning: test set has only one class. ROC-AUC/PR-AUC set to 0.")

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            print(f"  Accuracy: {acc:.4f}")
            print(f"  ROC AUC:  {roc_auc:.4f}")
            print(f"  PR AUC:   {pr_auc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print("  Confusion Matrix:")
            if cm.shape == (2, 2):
                print(f"    TN: {cm[0,0]:4} | FP: {cm[0,1]:4}")
                print(f"    FN: {cm[1,0]:4} | TP: {cm[1,1]:4}")
            else:
                print(f"    {cm}")

            # Threshold tag safe for filenames
            threshold_tag = f"{t:.5f}".replace(".", "p")

            model_path = model_dir / f"{name}_severe_logreg_{threshold_tag}.pkl"
            preds_path = (
                results_dir / f"{name}_severe_2022_predictions_{threshold_tag}.csv"
            )

            # Save model
            joblib.dump(best_model, model_path)

            # Save predictions
            preds_df = test_df[
                ["fips_code", "year", "month", "day_of_year", pct_col]
            ].copy()
            preds_df["actual"] = y_test.values
            preds_df["predicted"] = y_pred
            preds_df["probability"] = y_proba
            preds_df["threshold_used"] = t
            preds_df.to_csv(preds_path, index=False)

            # Look up overall class balance for this threshold
            thr_row = threshold_results[threshold_results["threshold"] == t]
            if not thr_row.empty:
                thr_row = thr_row.iloc[0]
                severe_ratio = float(thr_row["severe_ratio"])
                severe_count = int(thr_row["severe"])
                non_severe_count = int(thr_row["non_severe"])
            else:
                severe_ratio = None
                severe_count = None
                non_severe_count = None

            county_threshold_models.append(
                {
                    "threshold": float(t),
                    "threshold_pct": f"{t*100:.4f}%",
                    "is_most_balanced_threshold": float(t)
                    == float(best_balance_threshold),
                    "train_size": int(len(train_df)),
                    "val_size": int(len(val_df)),
                    "test_size": int(len(test_df)),
                    "train_class_distribution": {
                        "non_severe": int((y_train == 0).sum()),
                        "severe": int((y_train == 1).sum()),
                    },
                    "test_class_distribution": {
                        "non_severe": int((y_test == 0).sum()),
                        "severe": int((y_test == 1).sum()),
                    },
                    "overall_class_distribution": {
                        "non_severe": non_severe_count,
                        "severe": severe_count,
                        "severe_ratio": severe_ratio,
                    },
                    "best_cv_score": float(search.best_score_),
                    "best_params": search.best_params_,
                    "test_metrics": {
                        "accuracy": float(acc),
                        "roc_auc": float(roc_auc),
                        "pr_auc": float(pr_auc),
                        "f1": float(f1),
                    },
                    "confusion_matrix": cm.tolist(),
                    "model_path": str(model_path),
                    "predictions_path": str(preds_path),
                }
            )

            # Track best F1 for this county
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = float(t)

        all_results[name] = {
            "county_fips": fips,
            "thresholds_tested": [float(t) for t in test_thresholds],
            "threshold_analysis": threshold_results.to_dict(orient="records"),
            "most_balanced_threshold": {
                "threshold": best_balance_threshold,
                "threshold_pct": f"{best_balance_threshold*100:.4f}%",
                "severe_ratio": float(threshold_results.loc[best_idx, "severe_ratio"]),
            },
            "best_f1_threshold": {
                "threshold": best_f1_threshold,
                "threshold_pct": (
                    f"{best_f1_threshold*100:.4f}%" if best_f1_threshold is not None else None
                ),
                "f1": float(best_f1) if best_f1_threshold is not None else None,
            },
            "models_by_threshold": county_threshold_models,
        }

    summary_path = results_dir / "miami_orange_severe_by_threshold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nProcessing Complete!")
    print(f"Models saved to: {model_dir}")
    print(f"Threshold-specific predictions & summary saved to: {results_dir}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
