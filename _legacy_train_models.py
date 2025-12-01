import pandas as pd
import numpy as np
import time
import json
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, leaguedashplayerstats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# --- CONFIGURATION ---
SEASONS = ['2023-24', '2024-25']
OUTPUT_FILE = 'player_models.json'

# --- 1. GET PLAYERS ---
def get_active_players(limit=None):
    print("🔎 Fetching active player list...")
    # Get all players who have played this season
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2024-25').get_data_frames()[0]
    
    # Sort by minutes to prioritize relevant players
    sorted_players = stats.sort_values('MIN', ascending=False)
    
    if limit:
        sorted_players = sorted_players.head(limit)
        
    return sorted_players[['PLAYER_ID', 'PLAYER_NAME']].to_dict('records')

# --- 2. GET DEFENSE MAP ---
def get_team_defense_map():
    print("🛡️ Fetching defensive stats...")
    time.sleep(1)
    from nba_api.stats.static import teams
    nba_teams = teams.get_teams()
    id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
    
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season='2024-25', measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]
    
    # Map ID to Abbr
    stats['TEAM_ABBREVIATION'] = stats['TEAM_ID'].map(id_to_abbr)
    return dict(zip(stats['TEAM_ABBREVIATION'], stats['DEF_RATING']))

# --- 3. TRAIN MODEL FOR ONE PLAYER ---
def train_player_model(player_id, player_name, def_map):
    all_logs = []
    
    try:
        for season in SEASONS:
            time.sleep(0.6) # Rate limit
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            
            if gamelog.empty: continue

            # Calculate Fantasy Points
            gamelog['Fantasy_PTS'] = (
                (gamelog['FGM'] * 2) + (gamelog['FGA'] * -1) +
                (gamelog['FTM'] * 1) + (gamelog['FTA'] * -1) +
                (gamelog['FG3M'] * 1) + (gamelog['OREB'] * 1) +
                (gamelog['REB'] * 1) + (gamelog['AST'] * 2) +
                (gamelog['STL'] * 4) + (gamelog['BLK'] * 4) +
                (gamelog['TOV'] * -2)
            )
            
            # Prepare Features
            gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'])
            gamelog = gamelog.sort_values('GAME_DATE')
            
            # Rolling Averages (Shifted to represent pre-game knowledge)
            gamelog['Last_3_Avg'] = gamelog['Fantasy_PTS'].shift(1).rolling(window=3).mean()
            
            # Matchup Info
            gamelog['Opponent'] = gamelog['MATCHUP'].apply(lambda x: x.split(' ')[-1])
            gamelog['Opp_DefRtg'] = gamelog['Opponent'].map(def_map)
            gamelog['Is_Home'] = gamelog['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
            
            all_logs.append(gamelog.dropna())
            
        if not all_logs: return None
        
        full_df = pd.concat(all_logs)
        if len(full_df) < 10: return None # Not enough data
        
        # Train
        X = full_df[['Last_3_Avg', 'Opp_DefRtg', 'Is_Home']]
        y = full_df['Fantasy_PTS']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate Error (MAE) to use as Standard Deviation
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        
        # Return Coefficients
        return {
            'name': player_name,
            'coef_last_3': model.coef_[0],
            'coef_opp_def': model.coef_[1],
            'coef_home': model.coef_[2],
            'intercept': model.intercept_,
            'mae': mae,
            'samples': len(full_df)
        }
        
    except Exception as e:
        # print(f"Error training {player_name}: {e}")
        return None

# --- 4. MAIN RUNNER ---
def run_training_pipeline():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Setup
    def_map = get_team_defense_map()
    # Limit to top 50 for testing, remove limit for full run
    players_list = get_active_players(limit=50) 
    
    models = {}
    
    print(f"🏋️ Training models for {len(players_list)} players...")
    
    # 2. Train Loop
    for p in tqdm(players_list):
        result = train_player_model(p['PLAYER_ID'], p['PLAYER_NAME'], def_map)
        if result:
            models[p['PLAYER_NAME']] = result
            
    # 3. Save
    print(f"\n✅ Training Complete. Saving {len(models)} models to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(models, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    run_training_pipeline()
