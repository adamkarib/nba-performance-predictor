"""
features.py
-----------
Cleans raw NBA game log data and engineers features for modeling.

Key transformations:
  - Parse game date and sort chronologically per player
  - Calculate 5-game rolling averages for key stats
  - Compute days of rest between games
  - Encode home vs. away games
"""

import pandas as pd
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

# Columns pulled from the raw nba_api game log
RAW_COLS = [
    "PLAYER_NAME", "SEASON", "GAME_DATE", "MATCHUP",
    "PTS", "REB", "AST", "MIN", "FGA", "FG_PCT",
    "FG3A", "FG3_PCT", "FTM", "FTA", "TOV", "PLUS_MINUS",
    "WL"
]

# Stats to compute rolling averages over
ROLLING_STATS = ["PTS", "REB", "AST", "MIN", "FGA", "FG_PCT"]
ROLLING_WINDOW = 5


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant columns, parse dates, and sort data per player.
    """
    # Keep only columns that exist in the dataframe
    available = [c for c in RAW_COLS if c in df.columns]
    df = df[available].copy()

    # Parse game date
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y", errors="coerce")

    # Convert MIN to float (raw format can be "32:00" string)
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")

    # Sort each player's games chronologically
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, compute rolling averages over the last N games.
    The rolling window is shifted by 1 so we only use past data (no leakage).
    """
    for stat in ROLLING_STATS:
        if stat not in df.columns:
            continue
        col_name = f"rolling_{stat.lower()}_{ROLLING_WINDOW}"
        df[col_name] = (
            df.groupby("PLAYER_NAME")[stat]
            .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        )
    return df


def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the number of days since the player's last game.
    First game of each player's sequence gets 3 days (league average).
    """
    df["days_rest"] = (
        df.groupby("PLAYER_NAME")["GAME_DATE"]
        .transform(lambda x: x.diff().dt.days)
        .fillna(3)
        .clip(upper=10)  # Cap at 10 (e.g. after All-Star break)
    )
    return df


def add_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode home game (1) vs. away game (0) from the MATCHUP column.
    Home games contain 'vs.', away games contain '@'.
    """
    if "MATCHUP" in df.columns:
        df["home_game"] = df["MATCHUP"].str.contains("vs.").astype(int)
    return df


def add_win_label(df: pd.DataFrame) -> pd.DataFrame:
    """Convert W/L string to binary integer."""
    if "WL" in df.columns:
        df["win"] = (df["WL"] == "W").astype(int)
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

# Final feature columns used in modeling
FEATURE_COLS = [
    f"rolling_{s.lower()}_{ROLLING_WINDOW}" for s in ROLLING_STATS
] + ["days_rest", "home_game"]

TARGET_COL = "PTS"


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature engineering pipeline.

    Returns
    -------
    X : pd.DataFrame — feature matrix
    y : pd.Series    — target (points scored)
    """
    df = clean_raw_data(df)
    df = add_rolling_features(df)
    df = add_days_rest(df)
    df = add_home_away(df)
    df = add_win_label(df)

    # Drop rows missing target or any feature
    required = FEATURE_COLS + [TARGET_COL]
    available_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=available_required)

    X = df[[c for c in FEATURE_COLS if c in df.columns]]
    y = df[TARGET_COL]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    return X, y


if __name__ == "__main__":
    # Quick test with dummy data
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data

    raw = load_data(use_cache=True)
    X, y = build_features(raw)
    print(X.head())
