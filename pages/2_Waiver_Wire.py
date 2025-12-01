import streamlit as st
import pandas as pd
import numpy as np
from app_utils import (
    get_avg_std_from_api, get_player_team_abbr, calculate_adjusted_games, 
    normalize_name, is_nba_player
)

st.set_page_config(page_title="Waiver Wire", layout="wide", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Waiver Wire Simulator")

if not st.session_state.get('stats_loaded', False):
    st.warning("⚠️ Please load player stats on the 'Matchup Simulation' page first!")
    st.stop()

st.write("See how a roster move impacts your win probability.")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    drop_players = st.multiselect(
        "Drop Players",
        options=[p['name'] for p in st.session_state.your_summary]
    )
    
with col2:
    add_players_input = st.text_area("Add Free Agents (One per line)", placeholder="Jordan Poole\nNaz Reid")
    
with col3:
    st.write("##") # Spacer
    simulate_btn = st.button("🚀 Simulate Transaction")

# Helper function for simulations (Simplified version of Page 1 logic)
def quick_sim(roster_summary, n=1000):
    wins = 0
    # Prepare team params
    your_means = np.array([p['avg'] for p in roster_summary])
    your_stds = np.array([p['std'] for p in roster_summary])
    your_games = np.array([p['adjusted_games'] for p in roster_summary])
    
    opp_summary = st.session_state.opp_summary
    opp_means = np.array([p['avg'] for p in opp_summary])
    opp_stds = np.array([p['std'] for p in opp_summary])
    opp_games = np.array([p['adjusted_games'] for p in opp_summary])
    
    # Current scores
    def norm(x): return x.lower().replace("'","").replace(" ","")
    team_scores_norm = {norm(k): v for k, v in st.session_state.team_scores.items()}
    your_curr = team_scores_norm.get(norm(st.session_state.your_team), 0.0)
    opp_curr = team_scores_norm.get(norm(st.session_state.opponent_team), 0.0)
    
    for _ in range(n):
        y_score = your_curr + np.sum(np.random.normal(your_means, your_stds) * your_games)
        o_score = opp_curr + np.sum(np.random.normal(opp_means, opp_stds) * opp_games)
        if y_score > o_score:
            wins += 1
    return (wins / n) * 100

if simulate_btn:
    # Parse adds
    add_player_names = [n.strip() for n in add_players_input.split('\n') if n.strip()]
    
    if not drop_players and not add_player_names:
        st.warning("Please select players to drop or add.")
    else:
        with st.spinner(f"Analyzing transaction..."):
            # 1. Get stats for new players
            new_players_data = []
            
            # Check if ML was used in the main load
            use_ml = st.session_state.get('use_ml_last_run', False)
            schedule_csv = st.session_state.get('schedule_csv')
            # Need target date from somewhere? 
            # Ideally we store 'target_date' in session state from Page 1
            # For now, let's assume user hasn't changed the date.
            # Or we can just use raw games if ML is on?
            
            # We need to recalculate games for new players.
            # This is tricky without the 'target_date' from Page 1.
            # Let's assume standard week for now or grab from session state if we saved it.
            # TODO: Save target_date in session_state in Page 1.
            
            # Fallback: Just use get_avg_std_from_api (API stats) for new players for speed
            # Unless we want to run ML for them too?
            # Let's stick to API stats for speed in Waiver Wire tool.
            
            for player_name in add_player_names:
                avg, std, _, _ = get_avg_std_from_api(player_name)
                if avg == 0:
                    st.error(f"Could not find stats for {player_name}")
                    continue
                    
                team_abbr = get_player_team_abbr(player_name)
                
                # We need games remaining.
                # Hack: Use a default target date (next Sunday) if not saved
                from datetime import datetime, timedelta
                today = datetime.now()
                days_until_sunday = (6 - today.weekday()) % 7
                next_sunday = today + timedelta(days=days_until_sunday)
                
                raw_games, adj_games, _ = calculate_adjusted_games(team_abbr, schedule_csv, next_sunday)
                
                final_games = raw_games if use_ml else adj_games
                
                new_players_data.append({
                    'name': player_name,
                    'avg': avg,
                    'std': std,
                    'adjusted_games': final_games
                })
            
            # Baseline
            baseline_prob = quick_sim(st.session_state.your_summary)
            
            # New Roster
            new_roster = [p.copy() for p in st.session_state.your_summary if p['name'] not in drop_players]
            new_roster.extend(new_players_data)
            
            new_prob = quick_sim(new_roster)
            
            diff = new_prob - baseline_prob
            
            st.metric("Win Probability Change", f"{diff:+.1f}%", f"{new_prob:.1f}% vs {baseline_prob:.1f}%")
            
            if diff > 0:
                st.balloons()
                st.success("✅ Recommended Move!")
            else:
                st.error("❌ This move decreases your chances.")
