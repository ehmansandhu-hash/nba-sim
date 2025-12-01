import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app_utils import (
    get_avg_std_from_api, get_player_team_abbr, calculate_adjusted_games, 
    normalize_name, is_nba_player, parse_bref_datetime, fix_abbr
)
from datetime import datetime, timedelta

st.set_page_config(page_title="Power Rankings", layout="wide", page_icon="🏆")

st.title("🏆 League Power Rankings")

if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Please load rosters on the Home page first!")
    st.stop()

if st.button("📊 Generate Power Rankings"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    rankings = []
    teams = list(st.session_state.team_rosters.keys())
    
    # Pre-load schedule
    schedule_csv = st.session_state.get('schedule_csv')
    try:
        sched_df = pd.read_csv(schedule_csv)
        sched_df['Home Team'] = sched_df['Home/Neutral'].apply(fix_abbr)
        sched_df['Away Team'] = sched_df['Visitor/Neutral'].apply(fix_abbr)
        sched_df['Date'] = sched_df.apply(parse_bref_datetime, axis=1)
    except:
        sched_df = pd.DataFrame()
        
    today = datetime.now()
    days_until_sunday = (6 - today.weekday()) % 7
    next_sunday = today + timedelta(days=days_until_sunday)
    
    for i, team_name in enumerate(teams):
        status_text.text(f"Analyzing {team_name}...")
        roster = st.session_state.team_rosters[team_name]
        total_proj = 0
        
        for player in roster:
            if not is_nba_player(player): continue
            
            # Check if we already have stats
            if player in st.session_state.get('player_stats', {}):
                avg, _ = st.session_state.player_stats[player]
            else:
                avg, _, _, _ = get_avg_std_from_api(player)
                
            if avg > 0:
                team_abbr = get_player_team_abbr(player)
                # Use raw games for simplicity in PR, or adjusted?
                # Let's use adjusted for fairness.
                _, adj, _ = calculate_adjusted_games(team_abbr, schedule_csv, next_sunday, sched_df)
                total_proj += avg * adj
                
        # Add current score
        def norm(x): return x.lower().replace("'","").replace(" ","")
        curr = st.session_state.team_scores.get(norm(team_name), 0.0)
        
        rankings.append({'Team': team_name, 'Projected Total': total_proj + curr, 'Current': curr, 'Remaining': total_proj})
        progress_bar.progress((i + 1) / len(teams))
        
    df = pd.DataFrame(rankings).sort_values('Projected Total', ascending=False).reset_index(drop=True)
    df.index += 1
    
    st.dataframe(df)
    
    fig = px.bar(df, x='Team', y='Projected Total', color='Projected Total', title="Projected Week Totals")
    st.plotly_chart(fig, use_container_width=True)
