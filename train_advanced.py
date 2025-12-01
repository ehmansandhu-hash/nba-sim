import pandas as pd
import numpy as np
import time
import joblib
import xgboost as xgb
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from datetime import datetime

# --- CONFIGURATION ---
SEASONS = ['2023-24', '2024-25']
OUTPUT_FILE = 'advanced_player_models.joblib'
DVP_FILE = 'dvp_stats.joblib'

# --- 1. DATA FETCHING ---
def get_all_game_logs():
    """Fetch game logs for ALL players to build DvP stats"""
    print("🌍 Fetching global game logs for DvP analysis...")
    all_logs = []
    
    # We iterate by season. 
    # Note: Fetching 'LeagueGameLog' might be faster but 'PlayerGameLog' gives us specific player columns easily.
    # Actually, 'LeagueDashPlayerStats' doesn't give game-by-game.
    # We will use a trick: Fetch top 300 players' logs. It's slow but accurate.
    # OR: Use 'LeagueGameLog' endpoint which gives all games.
    from nba_api.stats.endpoints import leaguegamelog
    
    for season in SEASONS:
        print(f"  - Fetching {season}...")
        time.sleep(1)
        logs = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P').get_data_frames()[0]
        logs['SEASON_ID'] = season
        all_logs.append(logs)
        
    return pd.concat(all_logs)

def build_dvp_map(global_logs, player_positions):
    """
    Build Defense vs Position Map.
    Returns: {(Opponent_Abbr, Position): Avg_FP_Allowed}
    """
    print("🛡️ Building Defense vs. Position (DvP) Map...")
    
    # Calculate FP for every row
    # Note: LeagueGameLog columns are slightly different? Let's check standard columns.
    # Usually: PTS, REB, AST, STL, BLK, TOV, FGM, FGA, FTM, FTA, FG3M
    
    # Standardize columns if needed
    if 'MATCHUP' not in global_logs.columns:
        return {}

    global_logs['Fantasy_PTS'] = (
        (global_logs['FGM'] * 2) + (global_logs['FGA'] * -1) +
        (global_logs['FTM'] * 1) + (global_logs['FTA'] * -1) +
        (global_logs['FG3M'] * 1) + (global_logs['OREB'] * 1) +
        (global_logs['REB'] * 1) + (global_logs['AST'] * 2) +
        (global_logs['STL'] * 4) + (global_logs['BLK'] * 4) +
        (global_logs['TOV'] * -2)
    )
    
    # Extract Opponent
    # Matchup format: "LAL vs. BOS" or "LAL @ BOS" -> Opponent is the last one
    global_logs['Opponent'] = global_logs['MATCHUP'].apply(lambda x: x.split(' ')[-1])
    
    # Merge Position
    # player_positions is dict {ID: Position}
    global_logs['Position'] = global_logs['PLAYER_ID'].map(player_positions)
    
    # Filter out unknown positions
    df = global_logs.dropna(subset=['Position'])
    
    # Group by Opponent + Position
    dvp = df.groupby(['Opponent', 'Position'])['Fantasy_PTS'].mean().to_dict()
    
    return dvp

    from nba_api.stats.endpoints import commonallplayers
    stats = commonallplayers.CommonAllPlayers(season='2024-25').get_data_frames()[0]
    # Columns: PERSON_ID, DISPLAY_FIRST_LAST, TEAM_ID, TEAM_CITY, TEAM_NAME, TEAM_ABBREVIATION, TEAM_CODE, ROSTERSTATUS, FROM_YEAR, TO_YEAR, PLAYERCODE, GAMES_PLAYED_FLAG, TEAM_SLUG
    # Wait, CommonAllPlayers might not have position either in the summary?
    # Let's check 'commonplayerinfo' loop? No, too slow.
    # 'playerindex'?
    # Actually, 'leaguedashplayerstats' DOES usually have it. Maybe I missed it.
    # Let's try 'leaguedashplayerbiostats'.
    # Or just 'commonteamroster' for all teams?
    
    # Let's try a different approach:
    # We can infer position from the 'global_logs' if we fetch it first? No.
    
    # Let's use 'commonallplayers' with 'IsOnlyCurrentSeason=1'?
    # Actually, let's just use a static map if possible, or fetch it from 'commonplayerinfo' for the top 50 players ONLY.
    # But we need it for DvP map (all players).
    
    # Let's try 'leaguedashplayerbiostats'.
    from nba_api.stats.endpoints import leaguedashplayerbiostats
    stats = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season='2024-25').get_data_frames()[0]
    # Columns: PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, AGE, PLAYER_HEIGHT, PLAYER_WEIGHT, COLLEGE, COUNTRY, DRAFT_YEAR, DRAFT_ROUND, DRAFT_NUMBER, GP, PTS, REB, AST, NET_RATING, OREB_PCT, DREB_PCT, USG_PCT, TS_PCT, AST_PCT
    # Still no position?
    
    # Okay, 'commonteamroster' is the most reliable.
    # We iterate through all teams.
    pass

def get_player_positions():
    print("📍 Fetching player positions (via Rosters)...")
    from nba_api.stats.static import teams
    from nba_api.stats.endpoints import commonteamroster
    
    nba_teams = teams.get_teams()
    pos_map = {}
    
    for t in tqdm(nba_teams):
        try:
            time.sleep(0.4)
            roster = commonteamroster.CommonTeamRoster(team_id=t['id'], season='2024-25').get_data_frames()[0]
            for _, row in roster.iterrows():
                pid = row['PLAYER_ID']
                pos = row['POSITION']
                
                # Simplify
                if 'G' in pos: simple = 'G'
                elif 'F' in pos: simple = 'F'
                elif 'C' in pos: simple = 'C'
                else: simple = 'G'
                
                pos_map[pid] = simple
        except:
            continue
            
    return pos_map

# --- 2. TRAINING ---
def train_player_model(player_id, player_name, player_pos, global_logs, dvp_map):
    try:
        # Filter logs for this player
        df = global_logs[global_logs['PLAYER_ID'] == player_id].copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE')
        
        if len(df) < 10: return None
        
        # --- FEATURE ENGINEERING ---
        
        # 1. Rest Days
        df['Rest_Days'] = df['GAME_DATE'].diff().dt.days - 1
        df['Rest_Days'] = df['Rest_Days'].fillna(2).clip(0, 5) # Default 2, max 5
        
        # 2. DvP (Defense vs Position)
        # Look up what the opponent allows to THIS player's position
        df['DvP_Rank'] = df['Opponent'].apply(lambda opp: dvp_map.get((opp, player_pos), 40.0))
        
        # 3. Home/Away
        df['Is_Home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        
        # 4. Rolling Stats (Shifted 1 to avoid leakage)
        df['Roll_Min'] = df['MIN'].shift(1).rolling(3).mean().fillna(method='bfill')
        df['Roll_FP'] = df['Fantasy_PTS'].shift(1).rolling(3).mean().fillna(method='bfill')
        
        df = df.dropna()
        
        if len(df) < 10: return None
        
        # --- TRAIN XGBOOST ---
        # Model 1: Predict Minutes
        X = df[['Rest_Days', 'DvP_Rank', 'Is_Home', 'Roll_Min', 'Roll_FP']]
        y_min = df['MIN']
        y_fp = df['Fantasy_PTS']
        
        # XGBoost Regressor
        model_min = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1)
        model_min.fit(X, y_min)
        
        # Model 2: Predict FP (Directly, using Minutes as a feature? Or FP/Min?)
        # Let's predict FP directly but include Predicted Minutes as a feature during inference?
        # Simpler: Just predict FP using the same features + Minutes?
        # No, during inference we don't know Minutes.
        # So: 
        # Step A: Predict Minutes using Pre-Game Features.
        # Step B: Predict FP using Pre-Game Features + Predicted Minutes.
        
        # Train Model B (FP) using ACTUAL Minutes (to learn the relationship)
        X_fp = X.copy()
        X_fp['Predicted_Min'] = df['MIN'] # Train with actual minutes
        model_fp = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1)
        model_fp.fit(X_fp, y_fp)
        
        # Calc Error
        preds = model_fp.predict(X_fp)
        mae = mean_absolute_error(y_fp, preds)
        
        return {
            'model_min': model_min,
            'model_fp': model_fp,
            'mae': mae,
            'pos': player_pos,
            'last_roll_min': df.iloc[-1]['Roll_Min'], # Store latest state
            'last_roll_fp': df.iloc[-1]['Roll_FP']
        }
        
    except Exception as e:
        # print(f"Error {player_name}: {e}")
        return None

# --- 3. MAIN ---
def run_training():
    print("🚀 Starting Advanced Training Pipeline...")
    
    # 1. Global Data
    positions = get_player_positions()
    global_logs = get_all_game_logs()
    dvp_map = build_dvp_map(global_logs, positions)
    
    # Save DvP Map for inference
    joblib.dump(dvp_map, DVP_FILE)
    
    # 2. Train Models
    # Get top 50 players for now (to save time), user can expand later
    # Actually, let's do top 50 active players
    active_ids = list(positions.keys())
    
    # Sort by total minutes in global logs to find top players
    top_players = global_logs.groupby('PLAYER_ID')['MIN'].sum().sort_values(ascending=False).head(50).index.tolist()
    
    models = {}
    print(f"🏋️ Training XGBoost models for {len(top_players)} players...")
    
    for pid in tqdm(top_players):
        # Find name
        name_row = global_logs[global_logs['PLAYER_ID'] == pid].iloc[0]
        pname = name_row['PLAYER_NAME']
        ppos = positions.get(pid, 'G')
        
        res = train_player_model(pid, pname, ppos, global_logs, dvp_map)
        if res:
            models[pname] = res
            
    # 3. Save
    print(f"✅ Saving {len(models)} models to {OUTPUT_FILE}...")
    joblib.dump(models, OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    run_training()
