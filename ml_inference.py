import joblib
import os
import pandas as pd
import streamlit as st
import xgboost as xgb

MODEL_FILE = 'advanced_player_models.joblib'
DVP_FILE = 'dvp_stats.joblib'

@st.cache_resource
def load_advanced_models():
    """Load pre-trained XGBoost models and DvP map"""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DVP_FILE):
        return {}, {}
    
    models = joblib.load(MODEL_FILE)
    dvp_map = joblib.load(DVP_FILE)
    return models, dvp_map

def predict_score_advanced(player_name, recent_min, recent_fp, opponent_abbr, is_home, rest_days=1):
    """
    Predict fantasy score using XGBoost (Minutes -> FP).
    """
    models, dvp_map = load_advanced_models()
    
    if player_name not in models:
        return None, None # No model found
        
    m = models[player_name]
    
    # 1. Prepare Features
    # Features: ['Rest_Days', 'DvP_Rank', 'Is_Home', 'Roll_Min', 'Roll_FP']
    
    # Lookup DvP
    # Default to 40.0 if unknown
    dvp_rank = dvp_map.get((opponent_abbr, m['pos']), 40.0)
    
    # Create DataFrame for prediction (XGBoost expects feature names)
    features = pd.DataFrame([{
        'Rest_Days': float(rest_days),
        'DvP_Rank': float(dvp_rank),
        'Is_Home': int(is_home),
        'Roll_Min': float(recent_min),
        'Roll_FP': float(recent_fp)
    }])
    
    # 2. Predict Minutes
    pred_min = m['model_min'].predict(features)[0]
    
    # 3. Predict FP (using Predicted Minutes)
    # The FP model was trained with 'Predicted_Min' as the last feature (which was actual min during training)
    features['Predicted_Min'] = pred_min
    
    pred_fp = m['model_fp'].predict(features)[0]
    
    return max(0, pred_fp), m['mae'], pred_min
