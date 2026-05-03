import pandas as pd
import lightgbm as lgb
import pickle
from datetime import datetime
import requests
import polyline
import logging
import os

# --- Configuration ---
# Get the absolute path of the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the model path relative to the script
MODEL_PATH = os.path.join(SCRIPT_DIR, 'lgbm_model.sav')

# This MUST be the same value used in train_model.py
LOCATION_PRECISION = 3 

# Features our new model expects. Must match train_model.py
FEATURE_COLUMNS = [
    'pickup_month', 
    'pickup_day', 
    'pickup_weekday', 
    'pickup_hour',
    'pickup_lat_zone',
    'pickup_lon_zone'
]

# --- Model Loading ---
try:
    logging.info(f"Loading LightGBM model from {MODEL_PATH}...")
    lgbm_model = pickle.load(open(MODEL_PATH, 'rb'))
    logging.info("LightGBM model loaded successfully.")
except FileNotFoundError:
    logging.error(f"CRITICAL: {MODEL_PATH} not found.")
    logging.error("This file is created by running 'train_model.py'.")
    lgbm_model = None
except Exception as e:
    logging.error(f"Error loading {MODEL_PATH}: {e}")
    lgbm_model = None

# --- Route Optimization Function (Unchanged) ---
def get_optimized_route_from_osrm(locations):
    """
    Gets a fast, optimized route from the OSRM API trip service.
    Returns the optimized order, duration, and route shape.
    """
    coords = ";".join([f"{loc['lng']},{loc['lat']}" for loc in locations])
    url = f"http://router.project-osrm.org/trip/v1/driving/{coords}?source=first&roundtrip=false&geometries=polyline&steps=false"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] == 'Ok':
            trip = data['trips'][0]
            route_geometry = polyline.decode(trip['geometry'])
            # The 'waypoints' list is at the top level
            optimized_indices = [wp['waypoint_index'] for wp in data['waypoints']]
            
            return {
                "optimized_indices": optimized_indices,
                "duration": trip['duration'] / 60,  # Convert seconds to minutes
                "route_geometry": route_geometry
            }
    except Exception as e:
        logging.error(f"OSRM Error: {e}")
        return None
    return None

# --- Demand Forecasting Function (MODIFIED for new model) ---
def predict_demand(lat, lng, datetime_str):
    """
    Predicts the demand (trip count) using the new LGBM model.
    """
    if lgbm_model is None:
        logging.warning("LGBM model not loaded. Returning 0.")
        return 0
        
    try:
        dt_obj = datetime.fromisoformat(datetime_str)

        # --- NEW: Create features to match the new model ---
        
        # 1. Create time features
        pickup_month = dt_obj.month
        pickup_day = dt_obj.day
        pickup_weekday = dt_obj.weekday
        pickup_hour = dt_obj.hour
        
        # 2. Create location zone features by rounding
        # This matches the logic from train_model.py
        pickup_lat_zone = round(lat, LOCATION_PRECISION)
        pickup_lon_zone = round(lng, LOCATION_PRECISION)

        # Create the single-row DataFrame for prediction
        data = {
            'pickup_month': [pickup_month],
            'pickup_day': [pickup_day],
            'pickup_weekday': [pickup_weekday],
            'pickup_hour': [pickup_hour],
            'pickup_lat_zone': [pickup_lat_zone],
            'pickup_lon_zone': [pickup_lon_zone]
        }
        
        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)

        # Convert columns to 'category' dtype to match training
        # This is critical for the model to work correctly
        for col in FEATURE_COLUMNS:
            df[col] = df[col].astype('category')

        # Make prediction
        prediction = lgbm_model.predict(df)
        
        # The model predicts a count. Round it and ensure it's not negative.
        demand_score = max(0, int(round(prediction[0])))
        
        return demand_score
        
    except Exception as e:
        logging.error(f"Error during LGBM prediction: {e}")
        return 0

