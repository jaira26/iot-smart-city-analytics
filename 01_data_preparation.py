import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# STEP 1: LOAD TAXI/TRAFFIC DATA

print("\nSTEP 1: Loading Traffic Data...")

try:
    # Load parquet file
    taxi_df = pd.read_parquet('data/yellow_tripdata_2024-01.parquet')
    print(f"Traffic data loaded: {taxi_df.shape[0]:,} rows x {taxi_df.shape[1]} columns")
    print(f"Memory usage: {taxi_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
except FileNotFoundError:
    print("Error: yellow_tripdata_2024-01.parquet not found in data/ folder")
    exit()

# Display first few rows
print("\nFirst 5 rows:")
print(taxi_df.head())

print("\nColumn names:")
print(taxi_df.columns.tolist())

print("\nData types:")
print(taxi_df.dtypes)

# Check for missing values
print("\nMissing values:")
missing = taxi_df.isnull().sum()
missing_pct = (missing / len(taxi_df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# STEP 2: CLEAN AND PROCESS TRAFFIC DATA

print("\n" + "="*70)
print("STEP 2: Processing Traffic Data...")
print("="*70)

# Remove rows with missing critical data
initial_count = len(taxi_df)
taxi_df = taxi_df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 
                                   'trip_distance', 'PULocationID'])
print(f"\nRemoved {initial_count - len(taxi_df):,} rows with missing critical data")
print(f"Remaining records: {len(taxi_df):,}")

# Convert datetime columns
taxi_df['pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'])
taxi_df['dropoff_datetime'] = pd.to_datetime(taxi_df['tpep_dropoff_datetime'])

# Calculate trip duration in minutes
taxi_df['trip_duration_minutes'] = (
    taxi_df['dropoff_datetime'] - taxi_df['pickup_datetime']
).dt.total_seconds() / 60

# Filter out unrealistic trips (duration < 1 min or > 180 min, distance > 100 miles)
taxi_df = taxi_df[
    (taxi_df['trip_duration_minutes'] >= 1) & 
    (taxi_df['trip_duration_minutes'] <= 180) &
    (taxi_df['trip_distance'] > 0) &
    (taxi_df['trip_distance'] <= 100)
]

print(f"After filtering unrealistic trips: {len(taxi_df):,} records")

# Calculate average speed (mph)
taxi_df['avg_speed_mph'] = (taxi_df['trip_distance'] / taxi_df['trip_duration_minutes']) * 60

# Extract time features
taxi_df['hour'] = taxi_df['pickup_datetime'].dt.hour
taxi_df['day_of_week'] = taxi_df['pickup_datetime'].dt.dayofweek
taxi_df['day_name'] = taxi_df['pickup_datetime'].dt.day_name()
taxi_df['date'] = taxi_df['pickup_datetime'].dt.date
taxi_df['is_weekend'] = taxi_df['day_of_week'].isin([5, 6]).astype(int)

# Create time period categories
def get_time_period(hour):
    if 6 <= hour < 10:
        return 'Morning Rush'
    elif 10 <= hour < 16:
        return 'Midday'
    elif 16 <= hour < 20:
        return 'Evening Rush'
    elif 20 <= hour < 24:
        return 'Night'
    else:
        return 'Late Night'

taxi_df['time_period'] = taxi_df['hour'].apply(get_time_period)

# Create congestion indicator based on speed
def get_congestion_level(speed):
    if speed < 10:
        return 'Heavy Congestion'
    elif 10 <= speed < 20:
        return 'Moderate Congestion'
    elif 20 <= speed < 30:
        return 'Light Congestion'
    else:
        return 'Free Flow'

taxi_df['congestion_level'] = taxi_df['avg_speed_mph'].apply(get_congestion_level)

print("\nTraffic data processing complete!")
print(f"Date range: {taxi_df['pickup_datetime'].min()} to {taxi_df['pickup_datetime'].max()}")


# STEP 3: LOAD AIR QUALITY DATA

print("\n" + "="*70)
print("STEP 3: Loading Air Quality Data...")
print("="*70)

try:
    air_df = pd.read_csv('data/nyc_air_quality.csv')
    print(f"Air quality data loaded: {air_df.shape[0]:,} rows x {air_df.shape[1]} columns")
except FileNotFoundError:
    print("Error: nyc_air_quality.csv not found in data/ folder")
    exit()

print("\nFirst 5 rows:")
print(air_df.head())

print("\nColumn names:")
print(air_df.columns.tolist())

# Basic info about air quality data
print("\nData summary:")
print(air_df.describe())


# STEP 4: TRAFFIC PATTERN ANALYSIS

print("\n" + "="*70)
print("STEP 4: Analyzing Traffic Patterns...")
print("="*70)

# Overall statistics
print("\nOVERALL TRAFFIC STATISTICS:")
print(f"Total trips analyzed: {len(taxi_df):,}")
print(f"Average trip distance: {taxi_df['trip_distance'].mean():.2f} miles")
print(f"Average trip duration: {taxi_df['trip_duration_minutes'].mean():.2f} minutes")
print(f"Average speed: {taxi_df['avg_speed_mph'].mean():.2f} mph")
print(f"Median speed: {taxi_df['avg_speed_mph'].median():.2f} mph")

# Speed statistics by time period
print("\nAVERAGE SPEED BY TIME PERIOD:")
time_period_stats = taxi_df.groupby('time_period')['avg_speed_mph'].agg(['mean', 'median', 'std'])
print(time_period_stats.round(2))

# Congestion analysis
print("\nCONGESTION DISTRIBUTION:")
congestion_counts = taxi_df['congestion_level'].value_counts()
congestion_pct = (congestion_counts / len(taxi_df) * 100).round(2)
congestion_summary = pd.DataFrame({
    'Count': congestion_counts,
    'Percentage': congestion_pct
})
print(congestion_summary)

# Hourly patterns
print("\nHOURLY TRAFFIC PATTERNS:")
hourly_stats = taxi_df.groupby('hour').agg({
    'avg_speed_mph': 'mean',
    'trip_distance': 'mean',
    'PULocationID': 'count'
}).rename(columns={'PULocationID': 'trip_count'})
print(hourly_stats.round(2))

# Weekday vs Weekend
print("\nWEEKDAY vs WEEKEND COMPARISON:")
weekday_avg_speed = taxi_df[taxi_df['is_weekend'] == 0]['avg_speed_mph'].mean()
weekend_avg_speed = taxi_df[taxi_df['is_weekend'] == 1]['avg_speed_mph'].mean()
print(f"Weekday average speed: {weekday_avg_speed:.2f} mph")
print(f"Weekend average speed: {weekend_avg_speed:.2f} mph")
print(f"Difference: {abs(weekend_avg_speed - weekday_avg_speed):.2f} mph")


# STEP 5: VISUALIZATIONS

print("\n" + "="*70)
print("STEP 5: Creating Visualizations...")
print("="*70)

plt.style.use('seaborn-v0_8-darkgrid')

# Figure 1: Hourly Traffic Volume
print("Creating Figure 1: Hourly traffic volume...")
fig, ax = plt.subplots(figsize=(12, 6))
hourly_volume = taxi_df.groupby('hour').size()
ax.bar(hourly_volume.index, hourly_volume.values, color='steelblue', alpha=0.8, edgecolor='black')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Number of Trips', fontsize=12)
ax.set_title('NYC Traffic Volume by Hour', fontsize=16, fontweight='bold')
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Figure 2: Average Speed by Hour
print("Creating Figure 2: Average speed by hour...")
fig, ax = plt.subplots(figsize=(12, 6))
hourly_speed = taxi_df.groupby('hour')['avg_speed_mph'].mean()
ax.plot(hourly_speed.index, hourly_speed.values, marker='o', linewidth=2, 
        markersize=8, color='crimson')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Average Speed (mph)', fontsize=12)
ax.set_title('Average Traffic Speed Throughout the Day', fontsize=16, fontweight='bold')
ax.set_xticks(range(24))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 3: Congestion Distribution
print("Creating Figure 3: Congestion distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
congestion_counts.plot(kind='bar', ax=ax, color='coral', alpha=0.8, edgecolor='black')
ax.set_xlabel('Congestion Level', fontsize=12)
ax.set_ylabel('Number of Trips', fontsize=12)
ax.set_title('Traffic Congestion Distribution', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Figure 4: Speed Distribution
print("Creating Figure 4: Speed distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(taxi_df['avg_speed_mph'], bins=50, color='green', alpha=0.7, edgecolor='black')
ax.axvline(taxi_df['avg_speed_mph'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {taxi_df["avg_speed_mph"].mean():.1f} mph')
ax.axvline(taxi_df['avg_speed_mph'].median(), color='blue', linestyle='--', 
           linewidth=2, label=f'Median: {taxi_df["avg_speed_mph"].median():.1f} mph')
ax.set_xlabel('Average Speed (mph)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Traffic Speeds', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 5: Weekday vs Weekend Speed Comparison
print("Creating Figure 5: Weekday vs weekend comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Weekday
weekday_data = taxi_df[taxi_df['is_weekend'] == 0]
weekday_hourly = weekday_data.groupby('hour')['avg_speed_mph'].mean()
axes[0].plot(weekday_hourly.index, weekday_hourly.values, marker='o', 
             linewidth=2, markersize=8, color='#2E86AB')
axes[0].set_xlabel('Hour', fontsize=12)
axes[0].set_ylabel('Average Speed (mph)', fontsize=12)
axes[0].set_title('Weekday Traffic Pattern', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(24))
axes[0].grid(True, alpha=0.3)

# Weekend
weekend_data = taxi_df[taxi_df['is_weekend'] == 1]
weekend_hourly = weekend_data.groupby('hour')['avg_speed_mph'].mean()
axes[1].plot(weekend_hourly.index, weekend_hourly.values, marker='o', 
             linewidth=2, markersize=8, color='#A23B72')
axes[1].set_xlabel('Hour', fontsize=12)
axes[1].set_ylabel('Average Speed (mph)', fontsize=12)
axes[1].set_title('Weekend Traffic Pattern', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(24))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# STEP 6: SAVE PROCESSED DATA

print("\n" + "="*70)
print("STEP 6: Saving Processed Data...")
print("="*70)

# Sample the data for faster processing in next steps (keep 100k records)
if len(taxi_df) > 100000:
    taxi_sample = taxi_df.sample(n=100000, random_state=42)
    print(f"Sampled {len(taxi_sample):,} records for analysis")
else:
    taxi_sample = taxi_df
    print(f"Using all {len(taxi_sample):,} records")

# Save processed data
taxi_sample.to_csv('data/processed_traffic_data.csv', index=False)
print("Saved: data/processed_traffic_data.csv")

# Save summary statistics
summary_stats = pd.DataFrame({
    'Metric': ['Total Trips', 'Avg Distance (mi)', 'Avg Duration (min)', 
               'Avg Speed (mph)', 'Weekday Avg Speed', 'Weekend Avg Speed'],
    'Value': [len(taxi_df), taxi_df['trip_distance'].mean(), 
              taxi_df['trip_duration_minutes'].mean(), taxi_df['avg_speed_mph'].mean(),
              weekday_avg_speed, weekend_avg_speed]
})
summary_stats.to_csv('data/traffic_summary_stats.csv', index=False)
print("Saved: data/traffic_summary_stats.csv")


# STEP 7: KEY INSIGHTS

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("\n1. TRAFFIC VOLUME PATTERNS:")
peak_hour = hourly_volume.idxmax()
print(f"   Peak traffic hour: {peak_hour}:00 ({hourly_volume.max():,} trips)")
low_hour = hourly_volume.idxmin()
print(f"   Lowest traffic hour: {low_hour}:00 ({hourly_volume.min():,} trips)")

print("\n2. SPEED & CONGESTION:")
slowest_hour = hourly_speed.idxmin()
fastest_hour = hourly_speed.idxmax()
print(f"   Slowest hour: {slowest_hour}:00 ({hourly_speed.min():.1f} mph)")
print(f"   Fastest hour: {fastest_hour}:00 ({hourly_speed.max():.1f} mph)")
heavy_congestion_pct = (len(taxi_df[taxi_df['congestion_level'] == 'Heavy Congestion']) / len(taxi_df) * 100)
print(f"   Heavy congestion: {heavy_congestion_pct:.1f}% of trips")

print("\n3. WEEKEND EFFECT:")
speed_improvement = ((weekend_avg_speed - weekday_avg_speed) / weekday_avg_speed * 100)
print(f"   Weekend speeds are {abs(speed_improvement):.1f}% {'higher' if speed_improvement > 0 else 'lower'} than weekdays")

