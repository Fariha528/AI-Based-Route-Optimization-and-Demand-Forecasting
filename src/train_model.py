import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import numpy as np
import glob
import os

# --- Configuration ---
# Get the absolute path of the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the data directory relative to the script
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "data/")

# --- NEW LOGIC: ---
# We are predicting 'trip_count' (i.e., demand), not 'trip_duration'.
TARGET_VARIABLE = 'trip_count'

# This is the precision for rounding lat/lon. 
# 3 decimals = ~110 meter area
# This MUST match the precision used in model_utils.py
LOCATION_PRECISION = 3

# Features our new model will use
FEATURE_COLUMNS = [
    'pickup_month', 
    'pickup_day', 
    'pickup_weekday', 
    'pickup_hour',
    'pickup_lat_zone', # We get this from the lookup file
    'pickup_lon_zone'  # We get this from the lookup file
]

# These features must be treated as categories by the model
CATEGORICAL_FEATURES = [
    'pickup_month',
    'pickup_day',
    'pickup_weekday',
    'pickup_hour',
    'pickup_lat_zone',
    'pickup_lon_zone'
]

def load_and_aggregate_data(directory_path):
    """
    Loads all .parquet files, merges with zone lookup, 
    and aggregates by zone/time to create a demand (trip_count) dataset.
    """
    print(f"Searching for .parquet files in {directory_path}...")
    parquet_files = glob.glob(os.path.join(directory_path, "*.parquet"))
    
    if not parquet_files:
        print(f"Error: No .parquet files found in {directory_path}.")
        return None, None

    # --- 1. Load the Taxi Zone Lookup File ---
    lookup_path = os.path.join(directory_path, "taxi_zone_lookup.csv")
    try:
        print(f"Loading Taxi Zone Lookup from {lookup_path}...")
        df_lookup = pd.read_csv(lookup_path)
        # We only need the ID and coordinates
        df_lookup = df_lookup[['LocationID', 'latitude', 'longitude']]
        print("Taxi Zone Lookup loaded successfully.")
    except FileNotFoundError:
        print(f"Error: 'taxi_zone_lookup.csv' not found in {directory_path}.")
        print("Please ensure this file is in your data/ folder.")
        return None, None
    except Exception as e:
        print(f"Error loading taxi_zone_lookup.csv: {e}")
        return None, None

    print(f"Found {len(parquet_files)} trip files. Loading...")
    
    df_list = []
    for filepath in parquet_files:
        try:
            # --- 2. Load ONLY the columns we need from the trip data ---
            # This is the fix for your data: we load 'PULocationID'
            df_part = pd.read_parquet(filepath, columns=['tpep_pickup_datetime', 'PULocationID'])
            df_list.append(df_part)
        except Exception as e:
            print(f"Warning: Could not read {filepath} or missing columns. Skipping. Error: {e}")
            
    if not df_list:
        print("Error: All trip files failed to load or were missing required columns.")
        return None, None
        
    df = pd.concat(df_list, ignore_index=True)
    print(f"Trip data loaded. Total rows: {len(df)}. Merging with zone data...")

    # --- 3. Merge Trip Data with Lookup Data ---
    # This matches each trip's 'PULocationID' with the 'LocationID' from the lookup
    # and adds the 'latitude' and 'longitude' columns to our dataframe.
    df = pd.merge(
        df, 
        df_lookup, 
        left_on='PULocationID', 
        right_on='LocationID',
        how='left'
    )
    df = df.drop(columns=['PULocationID', 'LocationID'])
    print("Merge complete. Starting aggregation...")

    # --- 4. Rename & Convert Time ---
    df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime'}, inplace=True)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # --- 5. Create Time Features ---
    dt = df['pickup_datetime'].dt
    df['pickup_month'] = dt.month
    df['pickup_day'] = dt.day
    df['pickup_weekday'] = dt.weekday
    df['pickup_hour'] = dt.hour
    
    # --- 6. Create Location "Zone" Features ---
    # (This uses the 'latitude' and 'longitude' columns we just merged)
    df['pickup_lat_zone'] = df['latitude'].round(LOCATION_PRECISION)
    df['pickup_lon_zone'] = df['longitude'].round(LOCATION_PRECISION)

    # --- 7. Aggregate to Create Demand (THE KEY STEP) ---
    print("Aggregating data by zone and hour... This may take a moment.")
    
    agg_columns = [
        'pickup_month', 'pickup_day', 'pickup_weekday', 'pickup_hour',
        'pickup_lat_zone', 'pickup_lon_zone'
    ]
    
    # Group by our features and count the number of trips
    df_demand = df.groupby(agg_columns).size().reset_index(name=TARGET_VARIABLE)
    
    # Filter out impossible locations (lat/lon 0.0 from bad lookups)
    df_demand = df_demand[
        (df_demand['pickup_lat_zone'] != 0.0) & (df_demand['pickup_lon_zone'] != 0.0)
    ]

    if df_demand.empty:
        print("Error: Aggregation resulted in an empty dataset. Check data quality.")
        return None, None

    print(f"Aggregation complete. New dataset size: {len(df_demand)} rows.")

    # --- 8. Finalize X and y ---
    X = df_demand[FEATURE_COLUMNS]
    y = df_demand[TARGET_VARIABLE]
    
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype('category')
        
    return X, y

def train_lgbm_model(X, y):
    """Trains and returns a LightGBM model."""
    print("Splitting data for training and validation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Initializing LightGBM Regressor for predicting counts (demand)...")
    lgbm = lgb.LGBMRegressor(
        objective='poisson', # Poisson is well-suited for count data (like trip counts)
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        seed=42
    )

    print("Starting model training...")
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=100)],
        categorical_feature=CATEGORICAL_FEATURES
    )

    print("\n--- Model Evaluation Report (Predicting Trip Counts) ---")
    preds = lgbm.predict(X_test)
    preds[preds < 0] = 0 # Predictions can't be negative
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    medae = median_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} trips")
    print(f"         (Typical error in trip count)")
    print(f"   MAE (Mean Absolute Error):     {mae:.2f} trips")
    print(f"         (Average error in trip count)")
    print(f" MedAE (Median Absolute Error):   {medae:.2f} trips")
    print(f"         (The 'middle' error value)")
    print(f"      R² (Coefficient of Det.):  {r2:.3f}")
    print(f"         (Proportion of variance explained, 1.0 is perfect)")
    print("-----------------------------------")
    
    print(f"\nModel training complete.")
    
    return lgbm

def save_model(model, filename="lgbm_model.sav"):
    """Saves the trained model to a file using pickle."""
    # Save model relative to the script's directory
    save_path = os.path.join(SCRIPT_DIR, filename)
    print(f"Saving model to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

if __name__ == "__main__":
    X, y = load_and_aggregate_data(RAW_DATA_DIR)
    
    if X is not None and y is not None:
        model = train_lgbm_model(X, y)
        save_model(model)
        print("\n--- SUCCESS ---")
        print(f"The file 'lgbm_model.sav' has been created.")
        print("You can now run the Flask 'app.py' server.")

