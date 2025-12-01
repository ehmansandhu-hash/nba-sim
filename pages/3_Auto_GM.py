import streamlit as st
import pandas as pd
import numpy as np
from app_utils import (
    get_avg_std_from_api, get_player_team_abbr, calculate_adjusted_games, 
    normalize_name, is_nba_player
)
from datetime import datetime, timedelta

st.set_page_config(page_title="Auto-GM", layout="wide", page_icon="🤖")

st.title("🤖 Auto-GM: Find the Best Pickup")

if not st.session_state.get('stats_loaded', False):
    st.warning("⚠️ Please load player stats on the 'Matchup Simulation' page first!")
    st.stop()

st.write("I will scan the top free agents and tell you who adds the most Win %.")

# Inputs
drop_candidate = st.selectbox("Who would you drop?", [p['name'] for p in st.session_state.your_summary])
num_candidates = st.slider("How many free agents to scan?", 5, 50, 10)

if st.button("🤖 Run Auto-GM"):
    # Get Free Agents (from session state populated in Page 1)
    free_agents = st.session_state.get('free_agents', [])
    
    if not free_agents:
        st.error("No free agents found. Did you load rosters?")
        st.stop()
        
    # Filter top N by... well we don't have their stats yet.
    # We just take the first N from the list (which is sorted alphabetically? No, ideally sorted by %Owned or something)
    # ESPN scraping didn't give us %Owned.
    # We'll just take the first N for now.
    candidates = free_agents[:num_candidates]
    
    results = []
    progress_bar = st.progress(0)
    
    # Baseline
    # Helper sim function (duplicated from Page 2, ideally in utils but needs session state)
    def quick_sim(roster_summary, n=500): # Lower N for speed
        wins = 0
        your_means = np.array([p['avg'] for p in roster_summary])
        your_stds = np.array([p['std'] for p in roster_summary])
        your_games = np.array([p['adjusted_games'] for p in roster_summary])
        
        opp_summary = st.session_state.opp_summary
        opp_means = np.array([p['avg'] for p in opp_summary])
        opp_stds = np.array([p['std'] for p in opp_summary])
        opp_games = np.array([p['adjusted_games'] for p in opp_summary])
        
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

    baseline_prob = quick_sim(st.session_state.your_summary)
    
    use_ml = st.session_state.get('use_ml_last_run', False)
    schedule_csv = st.session_state.get('schedule_csv')
    today = datetime.now()
    days_until_sunday = (6 - today.weekday()) % 7
    next_sunday = today + timedelta(days=days_until_sunday)
    
    for i, fa in enumerate(candidates):
        # Get stats
        avg, std, _, _ = get_avg_std_from_api(fa)
        if avg == 0: continue
        
        team_abbr = get_player_team_abbr(fa)
        raw, adj, _ = calculate_adjusted_games(team_abbr, schedule_csv, next_sunday)
        final_games = raw if use_ml else adj
        
        # Sim
        new_roster = [p.copy() for p in st.session_state.your_summary if p['name'] != drop_candidate]
        new_roster.append({'name': fa, 'avg': avg, 'std': std, 'adjusted_games': final_games})
        
        prob = quick_sim(new_roster)
        diff = prob - baseline_prob
        
        results.append({'Player': fa, 'Win % Change': diff, 'New Win %': prob, 'Avg': avg, 'Games': final_games})
        progress_bar.progress((i + 1) / len(candidates))
        
    # Display
    res_df = pd.DataFrame(results).sort_values('Win % Change', ascending=False)
    st.dataframe(res_df.style.format({'Win % Change': '{:+.1f}%', 'New Win %': '{:.1f}%', 'Avg': '{:.1f}'}))
    
    best = res_df.iloc[0]
    if best['Win % Change'] > 0:
        st.success(f"🏆 Best Move: Add **{best['Player']}** (+{best['Win % Change']:.1f}%)")
    else:
        st.info("No players found that improve your team.")
