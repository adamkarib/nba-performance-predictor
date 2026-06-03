"""
main.py
-------
Entry point for the NBA Player Points Predictor.

Runs the full pipeline:
  1. Load data (from cache or NBA API)
  2. Engineer features
  3. Train and evaluate the model
  4. Save visualizations to /data

Usage:
    python main.py                 # use cached data if available
    python main.py --refresh       # force re-fetch from the NBA API
"""

import sys
import os
import argparse

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_data
from features import build_features
from model import run_model_pipeline


def main():
    parser = argparse.ArgumentParser(description="NBA Player Points Predictor")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-fetch data from the NBA API instead of using cache."
    )
    args = parser.parse_args()

    print("\n" + "🏀" * 20)
    print("  NBA PLAYER POINTS PREDICTOR")
    print("🏀" * 20 + "\n")

    # 1. Load data
    print("[1/3] Loading data...")
    raw_df = load_data(use_cache=not args.refresh)

    # 2. Feature engineering
    print("\n[2/3] Engineering features...")
    X, y = build_features(raw_df)

    # 3. Train + evaluate
    print("\n[3/3] Training and evaluating model...")
    results = run_model_pipeline(X, y)

    print("Pipeline complete! Check the /data folder for output charts.\n")


if __name__ == "__main__":
    main()
