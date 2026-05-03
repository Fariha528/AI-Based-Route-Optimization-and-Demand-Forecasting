import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from datetime import datetime

# Load the hourly demand data
df = pd.read_csv('hourly_demand.csv')
df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])

# Sort by datetime
df = df.sort_values('pickup_hour').reset_index(drop=True)

# Extract features for prediction (same as in train_model.py)
df['pickup_month'] = df['pickup_hour'].dt.month
df['pickup_day'] = df['pickup_hour'].dt.day
df['pickup_weekday'] = df['pickup_hour'].dt.weekday
df['pickup_hour_of_day'] = df['pickup_hour'].dt.hour

# Note: You'll need to add pickup_lat_zone and pickup_lon_zone
# These should match the location you're predicting for
# For this example, I'll use placeholder values - replace with actual coordinates
LOCATION_PRECISION = 3
# Replace these with your actual location coordinates
DEFAULT_LAT = 40.758  # Example: NYC coordinates
DEFAULT_LON = -73.985

df['pickup_lat_zone'] = round(DEFAULT_LAT, LOCATION_PRECISION)
df['pickup_lon_zone'] = round(DEFAULT_LON, LOCATION_PRECISION)

# Prepare features (matching FEATURE_COLUMNS from train_model.py)
feature_columns = [
    'pickup_month',
    'pickup_day',
    'pickup_weekday',
    'pickup_hour_of_day',  # Note: using 'pickup_hour_of_day' instead of 'pickup_hour'
    'pickup_lat_zone',
    'pickup_lon_zone'
]

# Rename to match the model's expected feature names
df_features = df.copy()
df_features['pickup_hour'] = df_features['pickup_hour_of_day']
X = df_features[['pickup_month', 'pickup_day', 'pickup_weekday', 'pickup_hour', 'pickup_lat_zone', 'pickup_lon_zone']]

# Convert categorical features
categorical_features = ['pickup_month', 'pickup_day', 'pickup_weekday', 'pickup_hour', 'pickup_lat_zone', 'pickup_lon_zone']
for col in categorical_features:
    X[col] = X[col].astype('category')

# Load the trained model
try:
    with open('lgbm_model.sav', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")

    # Make predictions
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative predictions

    df['predictions'] = predictions
    df['actual'] = df['ride_count']

    # Calculate error metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(df['actual'], df['predictions']))
    mae = mean_absolute_error(df['actual'], df['predictions'])
    r2 = r2_score(df['actual'], df['predictions'])

    print(f"\nModel Performance Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.3f}")

    # Create comparison visualizations
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Plot 1: Time series comparison
    axes[0].plot(df['pickup_hour'], df['actual'], label='Actual Demand', alpha=0.7, linewidth=1.5, color='blue')
    axes[0].plot(df['pickup_hour'], df['predictions'], label='Predicted Demand', alpha=0.7, linewidth=1.5, color='red')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Ride Count')
    axes[0].set_title('Actual vs Predicted Demand Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Scatter plot
    axes[1].scatter(df['actual'], df['predictions'], alpha=0.5, s=20)
    max_val = max(df['actual'].max(), df['predictions'].max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    axes[1].set_xlabel('Actual Demand')
    axes[1].set_ylabel('Predicted Demand')
    axes[1].set_title('Scatter Plot: Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Residuals (errors)
    residuals = df['actual'] - df['predictions']
    axes[2].plot(df['pickup_hour'], residuals, color='green', alpha=0.6, linewidth=1)
    axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[2].fill_between(df['pickup_hour'], residuals, 0, alpha=0.3, color='green')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Residual (Actual - Predicted)')
    axes[2].set_title('Prediction Residuals Over Time')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("\nGraph saved as 'prediction_comparison.png'")

    # Create additional detailed plot for a subset of data
    fig2, ax = plt.subplots(figsize=(15, 6))

    # Show only first 168 hours (1 week) for better visibility
    subset_size = min(168, len(df))
    df_subset = df.head(subset_size)

    x_pos = np.arange(len(df_subset))
    width = 0.35

    ax.bar(x_pos - width/2, df_subset['actual'], width, label='Actual Demand', alpha=0.8, color='blue')
    ax.bar(x_pos + width/2, df_subset['predictions'], width, label='Predicted Demand', alpha=0.8, color='red')

    ax.set_xlabel('Time Index')
    ax.set_ylabel('Ride Count')
    ax.set_title(f'Detailed Comparison: First {subset_size} Hours')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('prediction_comparison_detailed.png', dpi=300, bbox_inches='tight')
    print("Detailed graph saved as 'prediction_comparison_detailed.png'")

    # Save comparison data to CSV
    comparison_df = df[['pickup_hour', 'actual', 'predictions']].copy()
    comparison_df['error'] = comparison_df['actual'] - comparison_df['predictions']
    comparison_df['absolute_error'] = np.abs(comparison_df['error'])
    comparison_df['percentage_error'] = (comparison_df['error'] / (comparison_df['actual'] + 1)) * 100

    comparison_df.to_csv('prediction_comparison_data.csv', index=False)
    print("Comparison data saved as 'prediction_comparison_data.csv'")

    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Average Actual Demand: {df['actual'].mean():.2f}")
    print(f"Average Predicted Demand: {df['predictions'].mean():.2f}")
    print(f"Max Actual Demand: {df['actual'].max():.0f}")
    print(f"Max Predicted Demand: {df['predictions'].max():.0f}")

except FileNotFoundError:
    print("Error: 'lgbm_model.sav' not found. Please train the model first using train_model.py")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()