
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# STEP 1: LOAD DATA

print("\nSTEP 1: Loading Data...")

try:
    df = pd.read_csv('data/traffic_with_anomaly_flags.csv')
    print(f"Traffic data loaded: {df.shape[0]:,} rows")
except FileNotFoundError:
    print("Error: traffic_with_anomaly_flags.csv not found")
    print("Please run 02_congestion_detection.py first")
    exit()

# Convert datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df = df.sort_values('pickup_datetime').reset_index(drop=True)

print(f"Date range: {df['pickup_datetime'].min()} to {df['pickup_datetime'].max()}")


# STEP 2: CREATE TIME-SERIES FEATURES

print("\nSTEP 2: Engineering Time-Series Features...")

# Create hourly aggregations
hourly_df = df.groupby([df['pickup_datetime'].dt.floor('H')]).agg({
    'avg_speed_mph': 'mean',
    'trip_duration_minutes': 'mean',
    'trip_distance': 'mean',
    'is_anomaly': 'sum',
    'PULocationID': 'count'
}).reset_index()

hourly_df.columns = ['datetime', 'avg_speed', 'avg_duration', 'avg_distance', 
                     'anomaly_count', 'trip_count']

# Extract time features
hourly_df['hour'] = hourly_df['datetime'].dt.hour
hourly_df['day_of_week'] = hourly_df['datetime'].dt.dayofweek
hourly_df['is_weekend'] = hourly_df['day_of_week'].isin([5, 6]).astype(int)
hourly_df['day_of_month'] = hourly_df['datetime'].dt.day

# Cyclical encoding for time features
hourly_df['hour_sin'] = np.sin(2 * np.pi * hourly_df['hour'] / 24)
hourly_df['hour_cos'] = np.cos(2 * np.pi * hourly_df['hour'] / 24)
hourly_df['dow_sin'] = np.sin(2 * np.pi * hourly_df['day_of_week'] / 7)
hourly_df['dow_cos'] = np.cos(2 * np.pi * hourly_df['day_of_week'] / 7)

# Lag features (previous values)
hourly_df['speed_lag_1h'] = hourly_df['avg_speed'].shift(1)
hourly_df['speed_lag_3h'] = hourly_df['avg_speed'].shift(3)
hourly_df['speed_lag_24h'] = hourly_df['avg_speed'].shift(24)

# Rolling statistics
hourly_df['speed_rolling_mean_6h'] = hourly_df['avg_speed'].rolling(window=6, min_periods=1).mean()
hourly_df['speed_rolling_std_6h'] = hourly_df['avg_speed'].rolling(window=6, min_periods=1).std()
hourly_df['speed_rolling_mean_24h'] = hourly_df['avg_speed'].rolling(window=24, min_periods=1).mean()

# Anomaly rate
hourly_df['anomaly_rate'] = hourly_df['anomaly_count'] / hourly_df['trip_count']

# Drop rows with NaN from lag features
hourly_df_clean = hourly_df.dropna().reset_index(drop=True)

print(f"Created time-series dataset: {len(hourly_df_clean):,} hourly records")
print(f"Features created: {hourly_df_clean.shape[1]}")


# STEP 3: TRAIN/TEST SPLIT

print("\nSTEP 3: Preparing Train/Test Split...")

# Use 80% for training, 20% for testing
split_idx = int(len(hourly_df_clean) * 0.8)

train = hourly_df_clean.iloc[:split_idx].copy()
test = hourly_df_clean.iloc[split_idx:].copy()

print(f"Training set: {len(train):,} hours")
print(f"   From: {train['datetime'].min()}")
print(f"   To: {train['datetime'].max()}")
print(f"\nTest set: {len(test):,} hours")
print(f"   From: {test['datetime'].min()}")
print(f"   To: {test['datetime'].max()}")

# STEP 4: BUILD SPEED PREDICTION MODEL

print("\nSTEP 4: Building Speed Prediction Model...")

# Define features
feature_cols = [
    'hour', 'day_of_week', 'is_weekend',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'speed_lag_1h', 'speed_lag_3h', 'speed_lag_24h',
    'speed_rolling_mean_6h', 'speed_rolling_std_6h', 'speed_rolling_mean_24h',
    'trip_count', 'avg_distance'
]

X_train = train[feature_cols]
y_train = train['avg_speed']

X_test = test[feature_cols]
y_test = test['avg_speed']

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Model trained successfully!")

# Make predictions
train_predictions = rf_model.predict(X_train)
test_predictions = rf_model.predict(X_test)


# STEP 5: EVALUATE MODEL

print("\nSTEP 5: Evaluating Model Performance...")

# Training metrics
train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
train_r2 = r2_score(y_train, train_predictions)

# Testing metrics
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)

print("\nTRAINING SET:")
print(f"   MAE: {train_mae:.4f} mph")
print(f"   RMSE: {train_rmse:.4f} mph")
print(f"   R² Score: {train_r2:.4f}")

print("\nTEST SET:")
print(f"   MAE: {test_mae:.4f} mph")
print(f"   RMSE: {test_rmse:.4f} mph")
print(f"   R² Score: {test_r2:.4f}")

# Calculate accuracy
mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100
print(f"\nMAPE: {mape:.2f}%")
print(f"Prediction Accuracy: {100 - mape:.2f}%")

# STEP 6: FEATURE IMPORTANCE

print("\nSTEP 6: Analyzing Feature Importance...")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES:")
print(feature_importance.head(10).to_string(index=False))

# STEP 7: CONGESTION PREDICTION

print("\nSTEP 7: Building Congestion Event Predictor...")

# Define congestion threshold (speed < 15 mph)
congestion_threshold = 15
train['is_congested'] = (train['avg_speed'] < congestion_threshold).astype(int)
test['is_congested'] = (test['avg_speed'] < congestion_threshold).astype(int)

# Train congestion classifier
from sklearn.ensemble import RandomForestClassifier

congestion_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Training congestion prediction model...")
congestion_model.fit(X_train, train['is_congested'])

# Predict congestion events
congestion_predictions = congestion_model.predict(X_test)
congestion_proba = congestion_model.predict_proba(X_test)[:, 1]

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

print("\nCONGESTION PREDICTION PERFORMANCE:")
print(classification_report(test['is_congested'], congestion_predictions, 
                           target_names=['Normal', 'Congested']))

# STEP 8: GENERATE FUTURE FORECAST

print("\nSTEP 8: Generating 24-Hour Forecast...")

# Create future timestamps
last_datetime = hourly_df_clean['datetime'].max()
future_datetimes = pd.date_range(
    start=last_datetime + pd.Timedelta(hours=1),
    periods=24,
    freq='H'
)

# Create future dataframe
future_df = pd.DataFrame({'datetime': future_datetimes})
future_df['hour'] = future_df['datetime'].dt.hour
future_df['day_of_week'] = future_df['datetime'].dt.dayofweek
future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

# Cyclical features
future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
future_df['dow_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
future_df['dow_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)

# Use last known values for lag features
last_speed = hourly_df_clean['avg_speed'].iloc[-1]
last_trip_count = hourly_df_clean['trip_count'].iloc[-1]
last_distance = hourly_df_clean['avg_distance'].iloc[-1]

future_df['speed_lag_1h'] = last_speed
future_df['speed_lag_3h'] = last_speed
future_df['speed_lag_24h'] = last_speed
future_df['speed_rolling_mean_6h'] = hourly_df_clean['speed_rolling_mean_6h'].iloc[-1]
future_df['speed_rolling_std_6h'] = hourly_df_clean['speed_rolling_std_6h'].iloc[-1]
future_df['speed_rolling_mean_24h'] = hourly_df_clean['speed_rolling_mean_24h'].iloc[-1]
future_df['trip_count'] = last_trip_count
future_df['avg_distance'] = last_distance

# Make predictions
X_future = future_df[feature_cols]
future_speed_predictions = rf_model.predict(X_future)
future_congestion_proba = congestion_model.predict_proba(X_future)[:, 1]

future_df['predicted_speed'] = future_speed_predictions
future_df['congestion_probability'] = future_congestion_proba

print(f"Generated forecast for next 24 hours")
print(f"Forecast period: {future_datetimes[0]} to {future_datetimes[-1]}")

# STEP 9: VISUALIZATIONS

print("\nSTEP 9: Creating Visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')

# Figure 1: Actual vs Predicted (Test Set)
print("Creating Figure 1: Test predictions...")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(test['datetime'], y_test.values, label='Actual', linewidth=1.5, alpha=0.8, color='blue')
ax.plot(test['datetime'], test_predictions, label='Predicted', linewidth=1.5, 
        alpha=0.8, color='red', linestyle='--')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Average Speed (mph)', fontsize=12)
ax.set_title(f'Traffic Speed Prediction (R² = {test_r2:.4f})', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Figure 2: Prediction Errors
print("Creating Figure 2: Prediction error distribution...")
test_errors = y_test - test_predictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(test_errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.set_xlabel('Prediction Error (mph)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prediction Errors', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 3: Feature Importance
print("Creating Figure 3: Feature importance...")
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance.head(12)
ax.barh(top_features['Feature'], top_features['Importance'], 
        color='teal', alpha=0.7, edgecolor='black')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Top 12 Most Important Features', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Figure 4: Actual vs Predicted Scatter
print("Creating Figure 4: Scatter plot...")
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(y_test, test_predictions, alpha=0.5, s=20, color='steelblue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Speed (mph)', fontsize=12)
ax.set_ylabel('Predicted Speed (mph)', fontsize=12)
ax.set_title(f'Actual vs Predicted Speed (R² = {test_r2:.4f})', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 5: 24-Hour Forecast
print("Creating Figure 5: Future forecast...")
fig, ax = plt.subplots(figsize=(15, 6))

# Plot last 48 hours of actual data
recent_data = hourly_df_clean.tail(48)
ax.plot(recent_data['datetime'], recent_data['avg_speed'], 
        label='Historical Data', linewidth=1.5, color='blue', alpha=0.7)

# Plot future forecast
ax.plot(future_df['datetime'], future_df['predicted_speed'], 
        label='24h Forecast', linewidth=2, color='red', linestyle='--', alpha=0.8)

ax.axvline(last_datetime, color='green', linestyle=':', linewidth=2, 
           label='Forecast Start')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Average Speed (mph)', fontsize=12)
ax.set_title('Traffic Speed Forecast: Next 24 Hours', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Figure 6: Congestion Probability Forecast
print("Creating Figure 6: Congestion probability forecast...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(future_df['hour'], future_df['congestion_probability'], 
       color='crimson', alpha=0.7, edgecolor='black')
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='50% Threshold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Congestion Probability', fontsize=12)
ax.set_title('24-Hour Congestion Risk Forecast', fontsize=16, fontweight='bold')
ax.set_xticks(range(24))
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# STEP 10: SAVE RESULTS

print("\nSTEP 10: Saving Results...")

# Save test predictions
test_results = pd.DataFrame({
    'datetime': test['datetime'],
    'actual_speed': y_test.values,
    'predicted_speed': test_predictions,
    'error': y_test.values - test_predictions,
    'actual_congestion': test['is_congested'],
    'predicted_congestion': congestion_predictions
})

test_results.to_csv('data/speed_prediction_results.csv', index=False)
print(f"Saved: data/speed_prediction_results.csv ({len(test_results):,} records)")

# Save future forecast
future_df[['datetime', 'hour', 'predicted_speed', 'congestion_probability']].to_csv(
    'data/24h_traffic_forecast.csv', index=False
)
print("Saved: data/24h_traffic_forecast.csv")

# Save feature importance
feature_importance.to_csv('data/model_feature_importance.csv', index=False)
print("Saved: data/model_feature_importance.csv")

# STEP 11: KEY INSIGHTS & RECOMMENDATIONS

print("\n" + "="*70)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n1. MODEL PERFORMANCE:")
print(f"   Speed prediction R²: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"   Average prediction error: {test_mae:.2f} mph")
print(f"   Prediction accuracy: {100 - mape:.2f}%")

print("\n2. KEY PREDICTORS:")
top_3_features = feature_importance['Feature'].head(3).tolist()
print(f"   Most important factors: {', '.join(top_3_features)}")

print("\n3. FORECAST INSIGHTS (Next 24h):")
peak_congestion_hour = future_df.loc[future_df['congestion_probability'].idxmax()]
min_speed_hour = future_df.loc[future_df['predicted_speed'].idxmin()]
print(f"   Highest congestion risk: {peak_congestion_hour['hour']}:00 ({peak_congestion_hour['congestion_probability']:.1%})")
print(f"   Slowest predicted speed: {min_speed_hour['hour']}:00 ({min_speed_hour['predicted_speed']:.1f} mph)")
print(f"   Average forecast speed: {future_df['predicted_speed'].mean():.1f} mph")

high_risk_hours = future_df[future_df['congestion_probability'] > 0.5]
if len(high_risk_hours) > 0:
    print(f"   High-risk hours: {len(high_risk_hours)} hours with >50% congestion probability")


