import pandas as pd
import time
import sys
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, leaguedashplayerstats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm # Progress bar

# --- CONFIGURATION ---
TOP_N_PLAYERS = 20  # Start small (e.g. 20) to test, then change to 150 or 200!
SEASONS = ['2023-24', '2024-25']

# --- 1. GET THE LIST OF RELEVANT PLAYERS ---
def get_top_fantasy_players(limit=50):
    print(f"🔎 Finding top {limit} active players...")
    # This endpoint gives us a summary of the current season
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2024-25').get_data_frames()[0]
    
    # Sort by 'MIN' (Minutes) or 'NBA_FANTASY_PTS' to get the best guys
    top_players = stats.sort_values('MIN', ascending=False).head(limit)
    
    # Return a list of dictionaries: [{'id': 123, 'name': 'LeBron'}, ...]
    return top_players[['PLAYER_ID', 'PLAYER_NAME']].to_dict('records')

# --- 2. GET DEFENSE DATA (Do this once to save time) ---
def get_team_defense_map():
    print("🛡️ Fetching defensive stats...")
    time.sleep(1)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season='2024-25', measure_type_nullable='Advanced'
    ).get_data_frames()[0]
    return dict(zip(stats['TEAM_ABBREVIATION'], stats['DEF_RATING']))

# --- 3. THE "BRAIN" (Same logic as before, wrapped in a function) ---
def predict_player(player_id, player_name, def_map):
    all_logs = []
    
    # Fetch logs for training
    try:
        for season in SEASONS:
            time.sleep(0.6) # Short pause to be safe
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            
            # --- CUSTOM FANTASY FORMULA ---
            gamelog['Fantasy_PTS'] = (
                (gamelog['FGM'] * 2) + (gamelog['FGA'] * -1) +
                (gamelog['FTM'] * 1) + (gamelog['FTA'] * -1) +
                (gamelog['FG3M'] * 1) + (gamelog['OREB'] * 1) +
                (gamelog['REB'] * 1) + (gamelog['AST'] * 2) +
                (gamelog['STL'] * 4) + (gamelog['BLK'] * 4) +
                (gamelog['TOV'] * -2)
            )
            
            # --- FEATURES ---
            gamelog['GAME_DATE'] = pd.to_datetime(gamelog['GAME_DATE'])
            gamelog = gamelog.sort_values('GAME_DATE')
            
            gamelog['Last_3_Avg'] = gamelog['Fantasy_PTS'].shift(1).rolling(window=3).mean()
            gamelog['Opponent'] = gamelog['MATCHUP'].apply(lambda x: x.split(' ')[-1])
            gamelog['Opp_DefRtg'] = gamelog['Opponent'].map(def_map)
            gamelog['Is_Home'] = gamelog['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
            
            all_logs.append(gamelog.dropna())
            
        full_df = pd.concat(all_logs)
        
        # If player has no data (rare), skip
        if full_df.empty: return None

        # --- TRAIN ---
        X = full_df[['Last_3_Avg', 'Opp_DefRtg', 'Is_Home']]
        y = full_df['Fantasy_PTS']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # --- PREDICT NEXT GAME (Hypothetical Matchup against Average Defense) ---
        # NOTE: In a real app, you would scrape the Schedule to find their ACTUAL next opponent.
        # For now, we predict against a "Average Defense" (Rating 115) at Home.
        last_row = full_df.iloc[-1]
        
        current_form = last_row['Last_3_Avg']
        prediction = model.predict([[current_form, 115.0, 1]])[0]
        
        return {
            'Player': player_name,
            'Pred_Fantasy_Pts': round(prediction, 1),
            'Last_3_Avg': round(current_form, 1),
            'Status': 'Hot' if prediction > last_row['Fantasy_PTS'].mean() else 'Cold'
        }

    except Exception as e:
        return None # Skip player if error

# --- 4. MAIN EXECUTION LOOP ---
def run_league_analysis():
    # A. Setup
    def_map = get_team_defense_map()
    player_list = get_top_fantasy_players(limit=TOP_N_PLAYERS)
    
    results = []
    
    print(f"\n🚀 Starting Analysis for {len(player_list)} players...")
    print("This might take a moment. Grab a coffee! ☕")
    
    # B. Loop with Progress Bar
    for p in tqdm(player_list):
        data = predict_player(p['PLAYER_ID'], p['PLAYER_NAME'], def_map)
        if data:
            results.append(data)
            
    # C. Save & Show
    final_df = pd.DataFrame(results).sort_values('Pred_Fantasy_Pts', ascending=False)
    
    print("\n--- 🏆 TOP PREDICTED PERFORMERS (Next Game) ---")
    print(final_df.head(10))
    
    final_df.to_csv("League_Predictions.csv", index=False)
    print("\n✅ Saved full results to 'League_Predictions.csv'")

# Run it!
run_league_analysis()