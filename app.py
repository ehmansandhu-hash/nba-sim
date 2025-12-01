import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from app_utils import normalize_team_name, is_nba_player

# Page config
st.set_page_config(page_title="Fantasy Basketball AI", layout="wide", page_icon="🏀")

st.title("🏀 Fantasy Basketball AI Assistant")
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'stats_loaded' not in st.session_state:
    st.session_state.stats_loaded = False
if 'team_rosters' not in st.session_state:
    st.session_state.team_rosters = {}
if 'player_stats' not in st.session_state:
    st.session_state.player_stats = {}
if 'team_scores' not in st.session_state:
    st.session_state.team_scores = {}

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    # league_id = st.text_input("ESPN League ID", value="1063982169")
    league_id = "1063982169" # Hardcoded for now or keep input if preferred, user didn't ask to remove this but let's keep it clean
    # Actually user only asked to remove schedule csv input.
    league_id = st.text_input("ESPN League ID", value="1063982169")
    season_id = st.text_input("Season ID", value="2025-26")
    # schedule_csv = st.text_input("Schedule CSV", value=r'C:\Users\ehman\Downloads\nba-2025-UTC.csv')
    schedule_csv = "nba-2025-UTC.csv"
    st.session_state.schedule_csv = schedule_csv # Save for other pages
    
    if st.button("🔄 Load ESPN Data", type="primary"):
        st.session_state.loading = True

# Main Logic
if st.session_state.get('loading', False):
    with st.spinner("Loading ESPN rosters and scores..."):
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            import os
            
            # Setup Driver
            # Check for system installed driver first (Streamlit Cloud/Linux)
            system_driver_paths = [
                "/usr/bin/chromedriver",
                "/usr/lib/chromium-browser/chromedriver",
                "/usr/bin/chromium-browser" 
            ]
            
            driver_path = None
            for p in system_driver_paths:
                if os.path.exists(p) and "chromium-browser" not in p: # exclude browser binary
                    driver_path = p
                    break
            
            if driver_path:
                service = Service(executable_path=driver_path)
            else:
                # Fallback to webdriver_manager (Local Windows/Mac)
                service = Service(ChromeDriverManager().install())
                
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Use existing profile if available (CRITICAL for ESPN login)
            # Note: In cloud deployment, this path won't exist, so it will fall back to standard headless session.
            # This means private leagues WON'T work in cloud unless cookies are handled differently.
            user_data_dir = r"C:\Users\ehman\chrome_selenium_profile"
            if os.path.exists(user_data_dir):
                chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
            
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.maximize_window()
            st.session_state.driver = driver
            
            espn_rosters_url = f"https://fantasy.espn.com/basketball/league/rosters?leagueId={league_id}"
            driver.get(espn_rosters_url)
            time.sleep(4)
            
            # Scrape Rosters
            team_tables_bs = []
            # Get all tables that look like rosters
            team_tables_selenium = driver.find_elements(By.CSS_SELECTOR, "table")
            for tbl in team_tables_selenium:
                team_tables_bs.append(BeautifulSoup(tbl.get_attribute('outerHTML'), 'html.parser'))
            
            # Find team names - try multiple selectors
            team_name_elems = driver.find_elements(By.CSS_SELECTOR, ".teamName")
            if not team_name_elems:
                team_name_elems = driver.find_elements(By.CSS_SELECTOR, ".TeamName")
            
            team_names = [elem.text.strip() for elem in team_name_elems if elem.text.strip()]
            
            rosters = {}
            # Match tables to teams
            # Note: ESPN usually lists teams in order, matching the tables
            for i, table in enumerate(team_tables_bs):
                if i >= len(team_names): break
                t_name = normalize_team_name(team_names[i])
                players = []
                
                # Robust player finding: iterate all rows
                rows = table.find_all("tr")
                for row in rows:
                    # Try specific selectors first
                    name_tag = row.find("div", class_="player-column__athlete")
                    if not name_tag:
                        name_tag = row.find("div", class_="PlayerColumn__Athlete")
                    
                    # If not found, look for any link that might be a player
                    if not name_tag:
                        links = row.find_all("a", class_="AnchorLink")
                        for link in links:
                            # Heuristic: Player links usually don't have "Team" or "Stats" in text
                            # and are often bold or main content
                            txt = link.get_text().strip()
                            if txt and len(txt) > 3 and "View" not in txt and "Player" not in txt:
                                # Check if it looks like a name (has space)
                                if " " in txt:
                                    name_tag = link
                                    break
                    
                    if name_tag:
                        p_name = name_tag.get_text().strip()
                        # Clean up name (remove injury status suffixes)
                        # ESPN often appends these directly to the name text
                        for suffix in ["DTD", "O", "SSPD", "IR", "OUT"]:
                            if p_name.endswith(suffix):
                                p_name = p_name[:-len(suffix)].strip()
                                
                        # Clean up name (sometimes includes position/team like "LeBron James SF - LAL")
                        # Usually ESPN puts name in a specific div, but if we grabbed a link it might be clean
                        # Just in case, take the first part if it looks like "Name Pos - Team"
                        # But standard ESPN roster links are usually just the name.
                        if p_name and p_name != "PLAYER":
                            players.append(p_name)
                            
                rosters[t_name] = players
                
            st.session_state.team_rosters = rosters
            
            # Scrape Scores
            scoreboard_url = f"https://fantasy.espn.com/basketball/league/scoreboard?leagueId={league_id}"
            driver.get(scoreboard_url)
            time.sleep(4)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            matchups = soup.find_all("div", class_="MatchupBox")
            
            scores = {}
            for m in matchups:
                teams_in_matchup = m.find_all("div", class_="ScoreCell__TeamName")
                score_vals = m.find_all("div", class_="ScoreCell__Score")
                
                if len(teams_in_matchup) == 2 and len(score_vals) == 2:
                    t1 = normalize_team_name(teams_in_matchup[0].get_text())
                    t2 = normalize_team_name(teams_in_matchup[1].get_text())
                    s1 = float(score_vals[0].get_text()) if score_vals[0].get_text() != '--' else 0.0
                    s2 = float(score_vals[1].get_text()) if score_vals[1].get_text() != '--' else 0.0
                    scores[t1.lower().replace(" ","").replace("'","")] = s1
                    scores[t2.lower().replace(" ","").replace("'","")] = s2
            
            st.session_state.team_scores = scores
            st.session_state.data_loaded = True
            st.session_state.loading = False
            driver.quit()
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.loading = False

if st.session_state.get('data_loaded', False):
    st.success(f"✅ Loaded {len(st.session_state.team_rosters)} teams!")
    
    col1, col2 = st.columns(2)
    with col1:
        your_team = st.selectbox("Your Team", list(st.session_state.team_rosters.keys()))
        st.session_state.your_team = your_team
        your_score_key = your_team.lower().replace(' ','').replace("'","")
        st.write(f"**Score:** {st.session_state.team_scores.get(your_score_key, 0)}")
        st.dataframe(pd.DataFrame(st.session_state.team_rosters[your_team], columns=["Player"]), hide_index=True)
        
    with col2:
        opponent_team = st.selectbox("Opponent", list(st.session_state.team_rosters.keys()), index=1)
        st.session_state.opponent_team = opponent_team
        opp_score_key = opponent_team.lower().replace(' ','').replace("'","")
        st.write(f"**Score:** {st.session_state.team_scores.get(opp_score_key, 0)}")
        st.dataframe(pd.DataFrame(st.session_state.team_rosters[opponent_team], columns=["Player"]), hide_index=True)
        
    st.info("👉 Go to **Matchup Simulation** in the sidebar to run predictions!")
