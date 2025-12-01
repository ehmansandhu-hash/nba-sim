from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

try:
    print("Fetching Opponent Stats...")
    stats = leaguedashteamstats.LeagueDashTeamStats(season='2024-25', measure_type_detailed_defense='Opponent')
    df = stats.get_data_frames()[0]
    
    target_cols = ['OPP_PTS', 'OPP_REB', 'OPP_AST', 'OPP_STL', 'OPP_BLK', 'OPP_TOV']
    found_cols = [c for c in df.columns if c in target_cols]
    
    print(f"Found Columns: {found_cols}")
    print(f"First row example: {df[found_cols].iloc[0].to_dict()}")
    
except Exception as e:
    print(f"Error: {e}")
