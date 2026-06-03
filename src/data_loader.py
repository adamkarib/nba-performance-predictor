"""
data_loader.py
--------------
Fetches per-game stats for a list of NBA players using the nba_api library
and caches the results as a CSV in the /data directory.
"""

import os
import time
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Players to include (add or remove names here)
TARGET_PLAYERS = [
    "LeBron James",
    "Stephen Curry",
    "Kevin Durant",
    "Nikola Jokic",
    "Luka Doncic",
    "Jayson Tatum",
    "Devin Booker",
    "Anthony Edwards",
]

SEASONS = ["2022-23", "2023-24"]  # Seasons to pull data for


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_player_id(name: str) -> int | None:
    """Return the NBA player ID for a given full name."""
    matches = players.find_players_by_full_name(name)
    if not matches:
        print(f"  [WARNING] Player not found: {name}")
        return None
    return matches[0]["id"]


def fetch_game_logs(player_id: int, player_name: str, season: str) -> pd.DataFrame:
    """Fetch per-game logs for a single player and season."""
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season"
        )
        df = logs.get_data_frames()[0]
        df["PLAYER_NAME"] = player_name
        df["SEASON"] = season
        return df
    except Exception as e:
        print(f"  [ERROR] Could not fetch {player_name} ({season}): {e}")
        return pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────────

def load_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load NBA game log data for all target players and seasons.

    Parameters
    ----------
    use_cache : bool
        If True and a cached CSV exists, load from disk instead of calling the API.

    Returns
    -------
    pd.DataFrame
        Combined game log data for all players.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "nba_game_logs.csv")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_csv(cache_path)

    print("Fetching data from NBA API...")
    all_logs = []

    for name in TARGET_PLAYERS:
        player_id = get_player_id(name)
        if player_id is None:
            continue

        for season in SEASONS:
            print(f"  Fetching {name} — {season}")
            df = fetch_game_logs(player_id, name, season)
            if not df.empty:
                all_logs.append(df)
            time.sleep(0.6)  # Respect rate limits

    if not all_logs:
        raise ValueError("No data was fetched. Check player names and API connectivity.")

    combined = pd.concat(all_logs, ignore_index=True)
    combined.to_csv(cache_path, index=False)
    print(f"\nData saved to {cache_path}  ({len(combined)} rows)")

    return combined


if __name__ == "__main__":
    df = load_data(use_cache=False)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Players: {df['PLAYER_NAME'].unique()}")
