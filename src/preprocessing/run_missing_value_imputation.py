"""
Command-line interface for missing value imputation.

Usage:
    python run_missing_value_imputation.py --input merged_weather_outages_2019_2024_keep_all.csv
    python run_missing_value_imputation.py --analyze-only
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from missing_value_handler import create_missing_value_handler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Handle missing values in weather-outage dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/merged_weather_outages_2019_2024_keep_all.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: <input>_imputed.csv)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze missing values without imputation",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--report", type=str, default=None, help="Path to save missingness report CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns\n")

    handler = create_missing_value_handler(verbose=not args.quiet)

    print("=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)
    report = handler.analyze(df)

    if args.report:
        report_path = Path(args.report)
        report.missing_by_column.to_csv(report_path, index=False)
        print(f"\nMissingness report saved to: {report_path}")

    if args.analyze_only:
        print("\nAnalysis complete. Exiting (--analyze-only flag set).")
        return

    print("\n" + "=" * 60)
    print("APPLYING IMPUTATION")
    print("=" * 60)
    df_imputed = handler.fit_transform(df)

    print("\n" + "=" * 60)
    print("POST-IMPUTATION ANALYSIS")
    print("=" * 60)
    report_after = handler.analyze(df_imputed)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            input_path.parent / f"{input_path.stem}_imputed{input_path.suffix}"
        )

    print(f"\nSaving imputed dataset to: {output_path}")
    df_imputed.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input rows:  {len(df):,}")
    print(f"Output rows: {len(df_imputed):,}")
    print(f"Missing before: {df.isnull().sum().sum():,}")
    print(f"Missing after:  {df_imputed.isnull().sum().sum():,}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
