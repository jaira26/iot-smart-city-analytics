# IoT Smart City Traffic Analytics Platform

A data analytics project that processes NYC taxi trip data to identify urban traffic congestion patterns and forecast 24-hour traffic speeds. The platform combines unsupervised clustering with a Random Forest predictive model to give city planners actionable intelligence for optimizing traffic signal timing and infrastructure planning.

---

## Overview

The project analyzes 100,000 plus NYC taxi trip records sourced from the NYC Taxi and Limousine Commission open data portal. Two analytical layers were built: a congestion pattern discovery layer using Isolation Forest and K-means clustering, and a predictive layer using Random Forest with engineered time-based features.

---

## Key Results

- Isolation Forest identified over 1,000 locations with unusual traffic patterns across the NYC road network
- K-means clustering categorized locations into 4 distinct congestion zones based on speed, volume, and time-of-day characteristics
- Engineered 15 plus time-based features capturing hourly patterns, day-of-week trends, and rolling averages
- Random Forest model achieved 85 percent or higher R-squared and 90 percent or higher accuracy for 24-hour traffic speed forecasting
- Outputs include congestion probability maps enabling data-driven signal timing optimization

---

## Tech Stack

| Component | Technology |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Anomaly Detection | Scikit-learn (Isolation Forest) |
| Clustering | Scikit-learn (K-means) |
| Forecasting | Scikit-learn (Random Forest) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Repository Structure

```
iot-smart-city-analytics/

README.md
data/
    nyc_taxi_sample.csv         # NYC TLC trip records (sample)
notebooks/
    01_data_preparation.ipynb   # Data loading, cleaning, and EDA
    02_congestion_detection.ipynb  # Isolation Forest and K-means clustering
    03_predictive_analytics.ipynb  # Random Forest forecasting pipeline

```

---

## Methodology

### Data Preprocessing
- Loaded 100,000 plus NYC taxi trip records with pickup/dropoff coordinates, timestamps, trip duration, and distance
- Cleaned invalid coordinates, zero-distance trips, and extreme outlier durations
- Aggregated trip-level data to location-hour level for modeling

### Congestion Pattern Discovery
- Applied Isolation Forest to detect locations with statistically unusual traffic behavior
- Flagged 1,000 plus anomalous locations for further investigation
- Used K-means clustering (k=4) to group locations into distinct congestion zone types:
  - High-density central zones
  - Airport and transit hub zones
  - Residential low-traffic zones
  - Mixed-use moderate congestion zones

### Traffic Speed Forecasting
- Engineered 15 plus features including hour of day, day of week, is-weekend flag, rolling 3-hour and 6-hour averages, and lag features
- Trained a Random Forest regressor on 80 percent of data, tested on remaining 20 percent
- Model outputs 24-hour ahead speed forecasts with congestion probability scores by zone

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/jaira26/iot-smart-city-analytics.git
cd iot-smart-city-analytics

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch notebooks
jupyter notebook
```

Run notebooks in order: 01 then 02 then 03.

---

## Data Source

NYC Taxi and Limousine Commission. (2025). TLC Trip Record Data. NYC Open Data. Retrieved from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

---

## Author

Jairaghavendra Sridhar
MS Data Analytics Engineering, Northeastern University
https://github.com/jaira26 | https://linkedin.com/in/jairaghavendrasridhar26
