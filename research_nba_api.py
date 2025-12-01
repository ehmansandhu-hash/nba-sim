from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

# Try to fetch League Dash Team Stats with 'Opponent' measure type if possible, 
# or just fetch Base and see if it has OPP columns.
# According to docs/experience, LeagueDashTeamStats has a 'measure_type_detailed_defense' parameter?
# Let's try fetching 'Base' first and checking columns.

print("Fetching LeagueDashTeamStats (Base)...")
try:
    stats = leaguedashteamstats.LeagueDashTeamStats(season='2024-25', measure_type_detailed_defense='Base')
    df = stats.get_data_frames()[0]
    print("Columns in Base:")
    print(df.columns.tolist())
    
    # Check for OPP columns
    opp_cols = [c for c in df.columns if 'OPP' in c]
    print(f"Opponent Columns: {opp_cols}")
except Exception as e:
    print(f"Error fetching Base: {e}")

print("\nFetching LeagueDashTeamStats (Opponent)...")
try:
    # 'Opponent' might be a valid measure_type_detailed_defense or just measure_type?
    # Let's try 'Opponent' as measure_type
    stats_opp = leaguedashteamstats.LeagueDashTeamStats(season='2024-25', measure_type_detailed_defense='Opponent')
    df_opp = stats_opp.get_data_frames()[0]
    print("Columns in Opponent:")
    print(df_opp.columns.tolist())
except Exception as e:
    print(f"Error fetching Opponent: {e}")

# Also check for 'Defense'
print("\nFetching LeagueDashTeamStats (Defense)...")
try:
    stats_def = leaguedashteamstats.LeagueDashTeamStats(season='2024-25', measure_type_detailed_defense='Defense')
    df_def = stats_def.get_data_frames()[0]
    print("Columns in Defense:")
    print(df_def.columns.tolist())
except Exception as e:
    print(f"Error fetching Defense: {e}")
