from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

try:
    print("Fetching Advanced Stats...")
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season='2024-25', measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]
    print("Columns:", stats.columns.tolist())
    print("First row:", stats.iloc[0])
except Exception as e:
    print("Error:", e)
