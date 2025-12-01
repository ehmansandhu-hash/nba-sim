import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from nba_api.stats.static import players as players_static
from ml_inference import predict_score_advanced, load_advanced_models

# --- CONSTANTS ---
NBA_TEAM_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

NAME_OVERRIDES = {
    "og anunoby": "anunoby ogugua",
}

# --- HELPER FUNCTIONS ---

def normalize_team_name(name):
    n = name.replace("'", "'").replace("'", "'").replace("`", "'")
    n = n.replace("'S", "'s")
    return n.strip()

def calculate_points(row):
    """Calculate fantasy points based on custom scoring"""
    return (
        2 * row['FGM'] - row['FGA'] + row['FTM'] - row['FTA'] + row['FG3M'] +
        row['OREB'] + row['REB'] + 2 * row['AST'] + 4 * row['STL'] +
        4 * row['BLK'] - 2 * row['TOV'] + row['PTS']
    )

def is_nba_player(name):
    return ("Team" not in name) and ("Free Agency" not in name)

def normalize_name(name):
    """Normalize player names for matching"""
    name = name.lower()
    name = re.sub(r'[^a-z ]', '', name)
    name = re.sub(r'\b(jr|sr|ii|iii|iv|v|vi)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Apply overrides
    if name in NAME_OVERRIDES:
        name = NAME_OVERRIDES[name]
    
    return name

def find_player_by_name(player_name):
    """Find player using NBA API with fuzzy matching"""
    # Try exact match first
    player_list = players_static.find_players_by_full_name(player_name)
    if player_list:
        return player_list[0]
    
    # Try partial matches
    all_players = players_static.get_players()
    normalized_search = normalize_name(player_name)
    
    for player in all_players:
        normalized_full = normalize_name(player['full_name'])
        if normalized_search == normalized_full:
            return player
    
    # Try last name match
    search_last = player_name.split()[-1].lower()
    for player in all_players:
        if search_last in player['full_name'].lower():
            return player
    
    return None

def get_avg_std_from_api(player_name, season='2025-26'):
    """Get player stats using NBA API"""
    try:
        player = find_player_by_name(player_name)
        if not player:
            # st.warning(f"Could not find player: {player_name}")
            return 0, 0, 0, 0
        
        # Get game logs
        time.sleep(0.6)  # Rate limiting
        gamelog = playergamelog.PlayerGameLog(
            player_id=player['id'],
            season=season
        )
        
        df = gamelog.get_data_frames()[0]
        
        if len(df) == 0:
            return 0, 0, 0, 0
        
        # Calculate fantasy points for each game
        scores = []
        minutes = []
        for _, row in df.iterrows():
            try:
                score = calculate_points(row)
                scores.append(score)
                minutes.append(row['MIN'])
            except:
                continue
        
        if len(scores) > 0:
            # Return Avg FP, Std FP, Avg Min (last 3 games for recent form)
            recent_fp = np.mean(scores[:3]) if len(scores) >= 3 else np.mean(scores)
            recent_min = np.mean(minutes[:3]) if len(minutes) >= 3 else np.mean(minutes)
            return np.mean(scores), np.std(scores), recent_fp, recent_min
        else:
            return 0, 0, 0, 0
            
    except Exception as e:
        # st.warning(f"Error getting stats for {player_name}: {str(e)}")
        return 0, 0, 0, 0

def get_player_team_abbr(player_name):
    """Get player's current NBA team abbreviation from most recent game"""
    try:
        player = find_player_by_name(player_name)
        if not player:
            return 'N/A'
        
        # Get recent game to find current team
        time.sleep(0.6)
        gamelog = playergamelog.PlayerGameLog(
            player_id=player['id'],
            season='2025-26'
        )
        df = gamelog.get_data_frames()[0]
        
        if len(df) > 0:
            # Get team abbreviation from most recent game
            matchup = df.iloc[0]['MATCHUP']
            # Format is like "LAL vs. BOS" or "LAL @ BOS"
            team_abbr = matchup.split()[0]
            return team_abbr
        
        return 'N/A'
    except:
        return 'N/A'

def fix_abbr(team):
    """Convert team name to abbreviation"""
    abbr_map = {name: abbr for name, abbr in NBA_TEAM_ABBR.items()}
    # Also try direct lookup if already abbreviated
    if team in abbr_map.values():
        return team
    return abbr_map.get(team, team)

def fetch_defense_factors(season='2024-25'):
    """Fetch opponent stats and calculate defense factors"""
    try:
        # Fetch Opponent Stats
        stats = leaguedashteamstats.LeagueDashTeamStats(season=season, measure_type_detailed_defense='Opponent')
        df = stats.get_data_frames()[0]
        
        # Calculate approximate Fantasy Points Allowed per game
        df['FP_Allowed'] = (
            df['OPP_PTS'] * 1 +
            df['OPP_REB'] * 1 +
            df['OPP_AST'] * 2 +
            df['OPP_STL'] * 4 +
            df['OPP_BLK'] * 4 -
            df['OPP_TOV'] * 2
        ) / df['GP']
        
        league_avg = df['FP_Allowed'].mean()
        
        factors = {}
        for _, row in df.iterrows():
            factor = row['FP_Allowed'] / league_avg
            factors[row['TEAM_NAME']] = factor
            
        return factors
    except Exception as e:
        st.warning(f"Error fetching defense stats: {e}")
        return {}

def fetch_defense_ratings(season='2024-25'):
    """Fetch raw Defensive Ratings for ML model"""
    try:
        from nba_api.stats.static import teams
        nba_teams = teams.get_teams()
        id_to_abbr = {t['id']: t['abbreviation'] for t in nba_teams}
        
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense='Advanced'
        ).get_data_frames()[0]
        
        stats['TEAM_ABBREVIATION'] = stats['TEAM_ID'].map(id_to_abbr)
        return dict(zip(stats['TEAM_ABBREVIATION'], stats['DEF_RATING']))
    except:
        return {}

def parse_bref_datetime(row):
    """Parse datetime from Basketball Reference CSV format"""
    try:
        if 'Start (ET)' in row and pd.notna(row['Start (ET)']):
            dt_str = f"{row['Date']} {row['Start (ET)']}"
            date_val = pd.to_datetime(dt_str, errors='coerce')
            if pd.notna(date_val):
                return date_val

        date_val = pd.to_datetime(row['Date'], errors='coerce')
        if pd.notna(date_val):
            return date_val
        
        return pd.to_datetime(row['Date'].split()[0] if isinstance(row['Date'], str) else row['Date'], errors='coerce')
    except:
        return pd.NaT

def calculate_adjusted_games(team_abbr, schedule_csv, target_date, sched_df=None):
    """Calculate raw games and SOS-adjusted games for a team"""
    try:
        if sched_df is None:
            sched_df = pd.read_csv(schedule_csv)
            sched_df['Home Team'] = sched_df['Home/Neutral'].apply(fix_abbr)
            sched_df['Away Team'] = sched_df['Visitor/Neutral'].apply(fix_abbr)
            sched_df['Date'] = sched_df.apply(parse_bref_datetime, axis=1)
        
        today = datetime.now()
        target_dt = datetime.combine(target_date, datetime.min.time())
        
        if not team_abbr or team_abbr == 'N/A':
            return 0, 0.0, []
            
        # Filter games for this team
        games_df = sched_df[
            ((sched_df['Home Team'] == team_abbr) | (sched_df['Away Team'] == team_abbr)) &
            (sched_df['Date'].notna())
        ]
        
        # Filter by date range
        games_in_range = games_df[
            (games_df['Date'] >= today) &
            (games_df['Date'].dt.date <= target_dt.date())
        ]
        
        raw_games = len(games_in_range)
        adjusted_games = 0.0
        
        defense_factors = st.session_state.get('defense_factors', {})
        
        # Reverse map for Abbr -> Name
        abbr_to_name = {v: k for k, v in NBA_TEAM_ABBR.items()}
        
        for _, row in games_in_range.iterrows():
            # Determine opponent
            if row['Home Team'] == team_abbr:
                opponent = row['Away Team']
            else:
                opponent = row['Home Team']
            
            opp_full_name = abbr_to_name.get(opponent, opponent)
            factor = defense_factors.get(opp_full_name, 1.0)
            adjusted_games += factor
            
        return raw_games, adjusted_games, [d.date() for d in games_in_range['Date']]
        
    except Exception as e:
        return 0, 0.0, []
