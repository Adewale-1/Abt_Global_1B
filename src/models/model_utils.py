"""
Utility functions for model training and evaluation.

I provide shared helper functions to ensure consistency across
training, evaluation, and visualization scripts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
from datetime import datetime


class ModelUtils:
    """
    I implement utility functions for the ML pipeline.
    This ensures consistent data handling across all model scripts.
    """

    def __init__(self, project_root: str = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.data_dir = self.project_root / "data" / "ml_ready"
        self.results_dir = self.project_root / "results"
        self.models_dir = self.project_root / "models" / "trained"

    def load_data_with_features(
        self, data_file: str = "merged_weather_outages_2019_2024_encoded.csv"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load the encoded dataset and selected features.

        return both the full dataset and the list of selected feature names
        so the model only uses the features identified during feature selection.

        Returns:
            Tuple of (dataframe, list of feature names)
        """
        print("Loading encoded dataset...")
        data_path = self.data_dir / data_file
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

        print("\nLoading selected features...")
        features_path = self.results_dir / "feature_selection" / "selected_features.csv"
        features_df = pd.read_csv(features_path)
        selected_features = features_df["feature"].tolist()
        print(f"Using {len(selected_features)} selected features")

        # validate that all selected features exist in the dataset
        missing_features = set(selected_features) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Selected features not found in dataset: {missing_features}"
            )

        return df, selected_features

    def temporal_split(
        self, df: pd.DataFrame, target_col: str = "any_out"
    ) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
    ]:
        """
        Split data temporally to prevent leakage from lag/rolling features.

        I use temporal split instead of random split because:
        - The data contains lag features (1-3 days)
        - Rolling window features (3, 7, 14 days)
        - Random split would leak future information into training

        Split strategy:
        - Train: 2019-2022 (4 years for learning patterns)
        - Validation: 2023 (1 year for tuning)
        - Test: 2024 (1 year for final evaluation)

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        print("\nPerforming temporal split...")

        # ensure 'year' column exists
        if "year" not in df.columns:
            raise ValueError("Dataset must have 'year' column for temporal split")

        # create the splits
        train_mask = df["year"] <= 2022
        val_mask = df["year"] == 2023
        test_mask = df["year"] == 2024

        train_df = df[train_mask]
        val_df = df[val_mask]
        test_df = df[test_mask]

        print(f"Train set: {len(train_df):,} rows (years 2019-2022)")
        print(
            f"  - No-outage: {(train_df[target_col] == 0).sum():,} ({(train_df[target_col] == 0).mean():.2%})"
        )
        print(
            f"  - Outage: {(train_df[target_col] == 1).sum():,} ({(train_df[target_col] == 1).mean():.2%})"
        )

        print(f"\nValidation set: {len(val_df):,} rows (year 2023)")
        print(
            f"  - No-outage: {(val_df[target_col] == 0).sum():,} ({(val_df[target_col] == 0).mean():.2%})"
        )
        print(
            f"  - Outage: {(val_df[target_col] == 1).sum():,} ({(val_df[target_col] == 1).mean():.2%})"
        )

        print(f"\nTest set: {len(test_df):,} rows (year 2024)")
        print(
            f"  - No-outage: {(test_df[target_col] == 0).sum():,} ({(test_df[target_col] == 0).mean():.2%})"
        )
        print(
            f"  - Outage: {(test_df[target_col] == 1).sum():,} ({(test_df[target_col] == 1).mean():.2%})"
        )

        return train_df, val_df, test_df

    def prepare_features_target(
        self, df: pd.DataFrame, feature_names: List[str], target_col: str = "any_out"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target from dataframe.

        I separate this into its own function to ensure consistent
        feature ordering across train/val/test sets.
        """
        X = df[feature_names].copy()
        y = df[target_col].copy()

        # check for any missing values that might have slipped through
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(
                f"Warning: Found {missing_count} missing values in features, filling with 0"
            )
            X = X.fillna(0)

        return X, y

    def save_model(
        self,
        model: Any,
        model_name: str,
        hyperparameters: Dict = None,
        training_info: Dict = None,
    ) -> None:
        """
        Save model with metadata for reproducibility.

        I save:
        - The trained model (pickle)
        - Hyperparameters (JSON)
        - Training metadata (JSON)
        """
        # ensure the models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # save the model
        model_path = self.models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")

        # save hyperparameters if provided
        if hyperparameters:
            params_path = self.models_dir / f"{model_name}_params.json"
            with open(params_path, "w") as f:
                json.dump(hyperparameters, f, indent=2)
            print(f"Hyperparameters saved to: {params_path}")

        # save training info if provided
        if training_info:
            info_path = self.models_dir / f"{model_name}_info.json"
            # add timestamp to training info
            training_info["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(info_path, "w") as f:
                json.dump(training_info, f, indent=2)
            print(f"Training info saved to: {info_path}")

    def load_model(self, model_name: str) -> Tuple[Any, Dict]:
        """
        Load a trained model and its hyperparameters.

        return both the model and its parameters for full context.
        """
        # load the model
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")

        # load hyperparameters if they exist
        params_path = self.models_dir / f"{model_name}_params.json"
        hyperparameters = {}
        if params_path.exists():
            with open(params_path, "r") as f:
                hyperparameters = json.load(f)
            print(f"Hyperparameters loaded from: {params_path}")

        return model, hyperparameters

    def save_results(
        self, results: str, filename: str, results_subdir: str = "models"
    ) -> None:
        """
        Save text results to file.

        use this for saving training logs, evaluation reports, etc.
        """
        results_path = self.results_dir / results_subdir
        results_path.mkdir(parents=True, exist_ok=True)

        output_file = results_path / filename
        with open(output_file, "w") as f:
            f.write(results)

        print(f"Results saved to: {output_file}")

    def log_results(self, message: str, log_file: str = None) -> None:
        """
        Log a message to console and optionally to file.

        provide consistent logging across all scripts.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        print(formatted_message)

        if log_file:
            log_path = self.results_dir / "models" / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(formatted_message + "\n")


def get_class_weights_info(y: pd.Series) -> Dict[str, float]:
    """
    Calculate and return class weight information.

    compute this to help understand how 'balanced' class weights
    will penalize each class differently.
    """
    n_samples = len(y)
    n_classes = y.nunique()

    class_counts = y.value_counts().sort_index()
    class_weights = {}

    for class_label, count in class_counts.items():
        # use the same formula as sklearn's 'balanced' mode
        weight = n_samples / (n_classes * count)
        class_weights[int(class_label)] = weight

    return {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "class_counts": class_counts.to_dict(),
        "class_weights": class_weights,
    }
