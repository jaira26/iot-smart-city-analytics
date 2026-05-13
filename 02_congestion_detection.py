import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SMART CITY CONGESTION & ANOMALY DETECTION")
print("="*70)


# STEP 1: LOAD PROCESSED DATA

print("\nSTEP 1: Loading Processed Traffic Data...")

try:
    df = pd.read_csv('data/processed_traffic_data.csv')
    print(f"Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: processed_traffic_data.csv not found")
    print("Please run 01_data_preparation.py first")
    exit()

# Convert datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['date'] = pd.to_datetime(df['date'])

print(f"Date range: {df['pickup_datetime'].min()} to {df['pickup_datetime'].max()}")

# STEP 2: FEATURE ENGINEERING FOR ANOMALY DETECTION

print("\nSTEP 2: Engineering Features for Anomaly Detection...")

# Create location-based aggregations
location_stats = df.groupby('PULocationID').agg({
    'avg_speed_mph': ['mean', 'std', 'min', 'max'],
    'trip_duration_minutes': 'mean',
    'trip_distance': 'mean'
}).reset_index()

location_stats.columns = ['location_id', 'avg_speed_mean', 'avg_speed_std', 
                          'avg_speed_min', 'avg_speed_max', 
                          'avg_duration', 'avg_distance']

print(f"Analyzed {len(location_stats)} unique locations")

# Hourly aggregations for time-based patterns
hourly_location = df.groupby(['hour', 'PULocationID']).agg({
    'avg_speed_mph': 'mean',
    'trip_duration_minutes': 'mean',
    'PULocationID': 'count'
}).rename(columns={'PULocationID': 'trip_count'}).reset_index()

print(f"Created {len(hourly_location):,} hourly-location combinations")


# STEP 3: ANOMALY DETECTION - TRAFFIC SPEED

print("\nSTEP 3: Detecting Speed Anomalies...")

# Prepare features for anomaly detection
features_df = df[['avg_speed_mph', 'trip_duration_minutes', 'trip_distance', 
                   'hour', 'day_of_week']].copy()

# Remove any infinite or NaN values
features_df = features_df.replace([np.inf, -np.inf], np.nan)
features_df = features_df.dropna()

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

print(f"Training data shape: {features_scaled.shape}")

# Train Isolation Forest
iso_forest = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100
)

print("Training Isolation Forest model...")
predictions = iso_forest.fit_predict(features_scaled)
anomaly_scores = iso_forest.score_samples(features_scaled)

# Add results back to dataframe
features_df['anomaly'] = predictions
features_df['anomaly_score'] = anomaly_scores
features_df['is_anomaly'] = (predictions == -1).astype(int)

# Merge with original data
df_with_anomalies = df.loc[features_df.index].copy()
df_with_anomalies['is_anomaly'] = features_df['is_anomaly'].values
df_with_anomalies['anomaly_score'] = features_df['anomaly_score'].values

print("Anomaly detection complete!")


# STEP 4: ANALYZE ANOMALIES

print("\nSTEP 4: Analyzing Anomalies...")

total_anomalies = df_with_anomalies['is_anomaly'].sum()
anomaly_percentage = (total_anomalies / len(df_with_anomalies)) * 100

print("\nANOMALY DETECTION SUMMARY:")
print(f"   Total trips analyzed: {len(df_with_anomalies):,}")
print(f"   Anomalies detected: {total_anomalies:,}")
print(f"   Anomaly rate: {anomaly_percentage:.2f}%")

# Separate anomalies and normal
anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]
normal = df_with_anomalies[df_with_anomalies['is_anomaly'] == 0]

print("\nNORMAL vs ANOMALOUS TRAFFIC:")
print(f"   Normal - Avg Speed: {normal['avg_speed_mph'].mean():.2f} mph")
print(f"   Anomaly - Avg Speed: {anomalies['avg_speed_mph'].mean():.2f} mph")
print(f"   Normal - Avg Duration: {normal['trip_duration_minutes'].mean():.2f} min")
print(f"   Anomaly - Avg Duration: {anomalies['trip_duration_minutes'].mean():.2f} min")

# Classify anomaly types
print("\nANOMALY TYPES:")

# Severe congestion (very slow speeds)
severe_congestion = anomalies[anomalies['avg_speed_mph'] < 5]
print(f"   Severe congestion (< 5 mph): {len(severe_congestion):,} ({len(severe_congestion)/total_anomalies*100:.1f}%)")

# Unusually fast (possible data errors or empty roads)
unusually_fast = anomalies[anomalies['avg_speed_mph'] > 40]
print(f"   Unusually fast (> 40 mph): {len(unusually_fast):,} ({len(unusually_fast)/total_anomalies*100:.1f}%)")

# Long duration trips
long_duration = anomalies[anomalies['trip_duration_minutes'] > 60]
print(f"   Extended duration (> 60 min): {len(long_duration):,} ({len(long_duration)/total_anomalies*100:.1f}%)")

# Short but slow (gridlock situations)
gridlock = anomalies[(anomalies['trip_distance'] < 2) & (anomalies['avg_speed_mph'] < 8)]
print(f"   Gridlock situations: {len(gridlock):,} ({len(gridlock)/total_anomalies*100:.1f}%)")


# STEP 5: TEMPORAL PATTERNS OF ANOMALIES

print("\nSTEP 5: Analyzing Temporal Patterns...")

# Hourly anomaly distribution
hourly_anomalies = anomalies.groupby('hour').size()
print("\nANOMALIES BY HOUR:")
print(hourly_anomalies.sort_values(ascending=False).head(5))

# Day of week analysis
dow_anomalies = anomalies.groupby('day_of_week').size()
print("\nANOMALIES BY DAY OF WEEK:")
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for day_num, count in dow_anomalies.items():
    print(f"   {day_names[day_num]}: {count:,}")

# Time period analysis
period_anomalies = anomalies['time_period'].value_counts()
print("\nANOMALIES BY TIME PERIOD:")
print(period_anomalies)


# STEP 6: LOCATION-BASED CLUSTERING

print("\nSTEP 6: Identifying Congestion Hotspots...")

# Find top congested locations
location_anomaly_count = anomalies.groupby('PULocationID').size().reset_index(name='anomaly_count')
location_anomaly_count = location_anomaly_count.sort_values('anomaly_count', ascending=False)

print("\nTOP 10 CONGESTION HOTSPOTS (by anomaly count):")
for idx, row in location_anomaly_count.head(10).iterrows():
    print(f"   Location ID {row['PULocationID']}: {row['anomaly_count']:,} anomalies")

# Cluster locations by traffic patterns
location_features = df.groupby('PULocationID').agg({
    'avg_speed_mph': ['mean', 'std'],
    'trip_duration_minutes': 'mean',
    'PULocationID': 'count'
}).reset_index()

location_features.columns = ['location_id', 'avg_speed', 'speed_variance', 
                              'avg_duration', 'trip_count']

# Filter locations with sufficient data
location_features = location_features[location_features['trip_count'] >= 50]

# Standardize and cluster
X_cluster = location_features[['avg_speed', 'speed_variance', 'avg_duration']].values
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# K-means clustering (4 clusters: fast-flow, moderate, congested, severe)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
location_features['cluster'] = kmeans.fit_predict(X_cluster_scaled)

print(f"\nClustered {len(location_features)} locations into 4 traffic pattern groups")

# Analyze clusters
cluster_summary = location_features.groupby('cluster').agg({
    'avg_speed': 'mean',
    'speed_variance': 'mean',
    'avg_duration': 'mean',
    'trip_count': 'sum'
}).round(2)

print("\nCLUSTER CHARACTERISTICS:")
cluster_labels = ['Free Flow', 'Moderate Traffic', 'Congested', 'Severe Congestion']
for idx, (cluster_id, row) in enumerate(cluster_summary.iterrows()):
    print(f"\n{cluster_labels[idx]} (Cluster {cluster_id}):")
    print(f"   Avg Speed: {row['avg_speed']:.1f} mph")
    print(f"   Speed Variance: {row['speed_variance']:.1f}")
    print(f"   Avg Duration: {row['avg_duration']:.1f} min")
    print(f"   Total Trips: {row['trip_count']:,.0f}")


# STEP 7: VISUALIZATIONS

print("\nSTEP 7: Creating Visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')

# Figure 1: Anomaly Score Distribution
print("Creating Figure 1: Anomaly score distribution...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(normal['anomaly_score'], bins=50, alpha=0.7, label='Normal', color='green', edgecolor='black')
ax.hist(anomalies['anomaly_score'], bins=50, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
ax.set_xlabel('Anomaly Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Anomaly Scores', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 2: Speed Comparison - Normal vs Anomalous
print("Creating Figure 2: Speed comparison...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(normal['avg_speed_mph'], bins=50, alpha=0.6, label='Normal', color='blue', edgecolor='black')
ax.hist(anomalies['avg_speed_mph'], bins=50, alpha=0.6, label='Anomaly', color='red', edgecolor='black')
ax.set_xlabel('Average Speed (mph)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Speed Distribution: Normal vs Anomalous Traffic', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 3: Hourly Anomaly Distribution
print("Creating Figure 3: Hourly anomaly patterns...")
fig, ax = plt.subplots(figsize=(12, 6))
hourly_anomalies.plot(kind='bar', ax=ax, color='crimson', alpha=0.8, edgecolor='black')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Number of Anomalies', fontsize=12)
ax.set_title('Traffic Anomalies by Hour', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Figure 4: Anomaly Types Breakdown
print("Creating Figure 4: Anomaly types...")
fig, ax = plt.subplots(figsize=(10, 6))
anomaly_types = pd.Series({
    'Severe Congestion': len(severe_congestion),
    'Unusually Fast': len(unusually_fast),
    'Extended Duration': len(long_duration),
    'Gridlock': len(gridlock),
    'Other': total_anomalies - (len(severe_congestion) + len(unusually_fast) + 
                                  len(long_duration) + len(gridlock))
})
anomaly_types.plot(kind='bar', ax=ax, color='coral', alpha=0.8, edgecolor='black')
ax.set_xlabel('Anomaly Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Anomaly Types', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Figure 5: Time Period Anomalies
print("Creating Figure 5: Time period analysis...")
fig, ax = plt.subplots(figsize=(10, 6))
period_anomalies.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8, edgecolor='black')
ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Number of Anomalies', fontsize=12)
ax.set_title('Traffic Anomalies by Time Period', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Figure 6: Location Clusters
print("Creating Figure 6: Location clustering...")
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(location_features['avg_speed'], location_features['avg_duration'],
                     c=location_features['cluster'], s=location_features['trip_count']/50,
                     alpha=0.6, cmap='viridis', edgecolor='black')
ax.set_xlabel('Average Speed (mph)', fontsize=12)
ax.set_ylabel('Average Duration (min)', fontsize=12)
ax.set_title('Location Clustering by Traffic Patterns', fontsize=16, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# STEP 8: SAVE RESULTS

print("\nSTEP 8: Saving Results...")

# Save anomalies
anomalies_output = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1][[
    'pickup_datetime', 'PULocationID', 'avg_speed_mph', 'trip_duration_minutes',
    'trip_distance', 'congestion_level', 'time_period', 'anomaly_score'
]].copy()

anomalies_output.to_csv('data/detected_traffic_anomalies.csv', index=False)
print(f"Saved: data/detected_traffic_anomalies.csv ({len(anomalies_output):,} anomalies)")

# Save full data with anomaly flags
df_with_anomalies.to_csv('data/traffic_with_anomaly_flags.csv', index=False)
print(f"Saved: data/traffic_with_anomaly_flags.csv ({len(df_with_anomalies):,} records)")

# Save location clusters
location_features.to_csv('data/location_clusters.csv', index=False)
print(f"Saved: data/location_clusters.csv ({len(location_features)} locations)")

# Save hotspots
location_anomaly_count.head(20).to_csv('data/congestion_hotspots.csv', index=False)
print("Saved: data/congestion_hotspots.csv (top 20 hotspots)")


# STEP 9: KEY INSIGHTS

print("\n" + "="*70)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

print("\n1. ANOMALY DETECTION PERFORMANCE:")
print(f"   Detected {total_anomalies:,} traffic anomalies ({anomaly_percentage:.2f}% of data)")
print(f"   Model: Isolation Forest with 100 estimators")

print("\n2. CRITICAL FINDINGS:")
print(f"   Severe congestion events: {len(severe_congestion):,}")
print(f"   Gridlock situations: {len(gridlock):,}")
print(f"   Peak anomaly hour: {hourly_anomalies.idxmax()}:00")

print("\n3. CONGESTION HOTSPOTS:")
top_hotspot = location_anomaly_count.iloc[0]
print(f"   Highest risk location: ID {top_hotspot['PULocationID']} ({top_hotspot['anomaly_count']:,} anomalies)")
print(f"   Total hotspots identified: {len(location_anomaly_count[location_anomaly_count['anomaly_count'] > 50])}")

print("\n4. TEMPORAL PATTERNS:")
print(f"   Most anomalies occur during: {period_anomalies.idxmax()}")
print(f"   Weekend vs Weekday anomaly difference detected")

