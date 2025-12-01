import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from app_utils import (
    get_avg_std_from_api, get_player_team_abbr, fix_abbr, 
    parse_bref_datetime, fetch_defense_ratings, calculate_adjusted_games,
    is_nba_player, normalize_name, NBA_TEAM_ABBR
)
from ml_inference import predict_score_advanced, load_advanced_models

st.set_page_config(page_title="Matchup Simulation", layout="wide", page_icon="🏀")

st.title("🏀 Matchup Simulation")

if not st.session_state.get('data_loaded', False):
    st.warning("⚠️ Please load rosters on the Home page first!")
    st.stop()

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    n_simulations = st.slider("Number of Simulations", 1000, 50000, 10000, 1000)
    use_ml = st.checkbox("Use AI Projections 🤖", value=True, help="Use trained Machine Learning models.")
    
    # Date Selection (Global or Local? Let's keep it here for now)
    today = datetime.now()
    days_until_sunday = (6 - today.weekday()) % 7
    next_sunday = today + timedelta(days=days_until_sunday)
    target_date = st.date_input("Week End Date", value=next_sunday)

# --- LOAD STATS LOGIC ---
st.subheader("📊 Player Statistics")

if st.button("🔄 Load/Refresh Player Stats", type="primary"):
    your_team = st.session_state.get('your_team')
    opponent_team = st.session_state.get('opponent_team')
    
    if not your_team or not opponent_team:
        st.error("Please select teams on the Home page.")
    else:
        your_players = st.session_state.team_rosters.get(your_team, [])
        opp_players = st.session_state.team_rosters.get(opponent_team, [])
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        player_stats = {}
        player_to_team = {}
        all_players = [p for p in your_players + opp_players if is_nba_player(p)]
        
        # Load ML data if enabled
        ml_ratings = {}
        if use_ml:
            status_text.text("Loading Advanced ML models...")
            load_advanced_models()
            
        # Load Schedule
        schedule_csv = st.session_state.get('schedule_csv', r'C:\Users\ehman\Downloads\nba-2025-UTC.csv')
        try:
            sched_df = pd.read_csv(schedule_csv)
            sched_df['Home Team'] = sched_df['Home/Neutral'].apply(fix_abbr)
            sched_df['Away Team'] = sched_df['Visitor/Neutral'].apply(fix_abbr)
            sched_df['Date'] = sched_df.apply(parse_bref_datetime, axis=1)
        except Exception as e:
            st.error(f"Error loading schedule: {e}")
            sched_df = pd.DataFrame()

        # Load stats loop
        for idx, name in enumerate(all_players):
            status_text.text(f"Loading stats for {name}... ({idx+1}/{len(all_players)})")
            
            avg, std, recent_fp, recent_min = get_avg_std_from_api(name)
            team_abbr = get_player_team_abbr(name)
            player_to_team[name] = team_abbr
            
            # ML Prediction
            if use_ml and avg > 0 and team_abbr != 'N/A' and not sched_df.empty:
                try:
                    today = datetime.now()
                    target_dt = datetime.combine(target_date, datetime.min.time())
                    
                    games_df = sched_df[
                        ((sched_df['Home Team'] == team_abbr) | (sched_df['Away Team'] == team_abbr)) &
                        (sched_df['Date'].notna())
                    ]
                    games_in_range = games_df[
                        (games_df['Date'] >= today) &
                        (games_df['Date'].dt.date <= target_dt.date())
                    ]
                    
                    if not games_in_range.empty:
                        predicted_scores = []
                        last_game_date = today - timedelta(days=2)
                        
                        for _, row in games_in_range.iterrows():
                            is_home = 1 if row['Home Team'] == team_abbr else 0
                            opponent = row['Away Team'] if is_home else row['Home Team']
                            
                            game_date = row['Date']
                            rest_days = (game_date - last_game_date).days - 1
                            rest_days = max(0, min(rest_days, 5))
                            last_game_date = game_date
                            
                            pred, mae, pred_min = predict_score_advanced(
                                name, recent_min, recent_fp, opponent, is_home, rest_days
                            )
                            
                            if pred is not None:
                                predicted_scores.append(pred)
                                std = mae 
                        
                        if predicted_scores:
                            avg = np.mean(predicted_scores)
                except Exception as e:
                    pass
            
            player_stats[name] = (avg, std)
            progress_bar.progress((idx + 1) / len(all_players))
            
        st.session_state.player_stats = player_stats
        st.session_state.player_to_team = player_to_team
        
        # Calculate Games Left
        status_text.text("Calculating games remaining...")
        player_games_left = {}
        player_adjusted_games = {}
        
        for name in all_players:
            team_abbr = player_to_team.get(name)
            raw, adj, dates = calculate_adjusted_games(team_abbr, schedule_csv, target_date, sched_df)
            player_games_left[name] = raw
            player_adjusted_games[name] = adj
            
        st.session_state.player_games_left = player_games_left
        st.session_state.player_adjusted_games = player_adjusted_games
        
        # Build Summaries
        your_summary = []
        for name in your_players:
            if not is_nba_player(name): continue
            avg, std = player_stats.get(name, (0, 0))
            games_left = player_games_left.get(name, 0) if avg > 0 else 0
            
            if use_ml:
                adj_games = games_left
            else:
                adj_games = player_adjusted_games.get(name, 0) if avg > 0 else 0
                
            your_summary.append({'name': name, 'avg': avg, 'std': std, 'games_left': games_left, 'adjusted_games': adj_games})
            
        opp_summary = []
        for name in opp_players:
            if not is_nba_player(name): continue
            avg, std = player_stats.get(name, (0, 0))
            games_left = player_games_left.get(name, 0) if avg > 0 else 0
            
            if use_ml:
                adj_games = games_left
            else:
                adj_games = player_adjusted_games.get(name, 0) if avg > 0 else 0
                
            opp_summary.append({'name': name, 'avg': avg, 'std': std, 'games_left': games_left, 'adjusted_games': adj_games})
            
        st.session_state.your_summary = your_summary
        st.session_state.opp_summary = opp_summary
        st.session_state.stats_loaded = True
        st.session_state.use_ml_last_run = use_ml # Track if ML was used
        
        status_text.success("✅ Stats Loaded!")
        st.rerun()

# --- SIMULATION LOGIC ---
if st.session_state.get('stats_loaded', False):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Team")
        df_your = pd.DataFrame(st.session_state.your_summary)
        if not df_your.empty:
            st.dataframe(df_your[['name', 'avg', 'adjusted_games']], hide_index=True)
            
    with col2:
        st.subheader("Opponent")
        df_opp = pd.DataFrame(st.session_state.opp_summary)
        if not df_opp.empty:
            st.dataframe(df_opp[['name', 'avg', 'adjusted_games']], hide_index=True)
            
    # --- CHARTS ---
    st.markdown("---")
    st.subheader("📈 Detailed Analysis")
    
    if not df_your.empty and not df_opp.empty:
        df_your['Team'] = 'You'
        df_opp['Team'] = 'Opponent'
        df_all = pd.concat([df_your, df_opp])
        
        # Calculate Projected Total
        df_all['Projected Total'] = df_all['avg'] * df_all['adjusted_games']
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # 1. Projected Total Points
            fig_proj = px.bar(
                df_all.sort_values('Projected Total', ascending=True), 
                x='Projected Total', y='name', color='Team', orientation='h',
                title="Projected Total Points by Player",
                color_discrete_map={'You': 'green', 'Opponent': 'red'},
                height=600
            )
            st.plotly_chart(fig_proj, use_container_width=True)
            
        with col_c2:
            # 2. Average FP
            fig_avg = px.bar(
                df_all.sort_values('avg', ascending=True), 
                x='avg', y='name', color='Team', orientation='h',
                title="Average Fantasy Points per Game",
                color_discrete_map={'You': 'green', 'Opponent': 'red'},
                height=600
            )
            st.plotly_chart(fig_avg, use_container_width=True)
            
    if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {n_simulations} simulations..."):
            # Prepare params
            your_means = np.array([p['avg'] for p in st.session_state.your_summary])
            your_stds = np.array([p['std'] for p in st.session_state.your_summary])
            your_games = np.array([p['adjusted_games'] for p in st.session_state.your_summary])
            
            opp_means = np.array([p['avg'] for p in st.session_state.opp_summary])
            opp_stds = np.array([p['std'] for p in st.session_state.opp_summary])
            opp_games = np.array([p['adjusted_games'] for p in st.session_state.opp_summary])
            
            # Current scores
            def norm(x): return x.lower().replace("'","").replace(" ","")
            team_scores_norm = {norm(k): v for k, v in st.session_state.team_scores.items()}
            your_curr = team_scores_norm.get(norm(st.session_state.your_team), 0.0)
            opp_curr = team_scores_norm.get(norm(st.session_state.opponent_team), 0.0)
            
            # Simulate
            your_sims = np.zeros(n_simulations)
            opp_sims = np.zeros(n_simulations)
            
            for i in range(n_simulations):
                your_sims[i] = your_curr + np.sum(np.random.normal(your_means, your_stds) * your_games)
                opp_sims[i] = opp_curr + np.sum(np.random.normal(opp_means, opp_stds) * opp_games)
                
            wins = np.sum(your_sims > opp_sims)
            win_prob = (wins / n_simulations) * 100
            
            st.markdown("---")
            st.metric("Win Probability", f"{win_prob:.1f}%", delta=None)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=your_sims, name='You', opacity=0.75, marker_color='green'))
            fig.add_trace(go.Histogram(x=opp_sims, name='Opponent', opacity=0.75, marker_color='red'))
            fig.update_layout(barmode='overlay', title="Score Distribution", xaxis_title="Total Score")
            st.plotly_chart(fig, use_container_width=True)
