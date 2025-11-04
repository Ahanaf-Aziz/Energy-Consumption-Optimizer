import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 1: GENERATING & PREPROCESSING ENERGY DATA")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for 1 year
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
n_samples = len(dates)

# Weather data with seasonal patterns
temperature = 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 365)) + np.random.normal(0, 2, n_samples)
humidity = 60 + 20 * np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 365)) + np.random.normal(0, 5, n_samples)

# Energy consumption with multiple patterns
base_load = 50  # Baseline consumption
time_of_day_pattern = 30 * (1 + np.sin(np.arange(n_samples) % 24 * 2 * np.pi / 24))
temperature_sensitivity = 5 * np.abs(temperature - 20)
random_variation = np.random.normal(0, 5, n_samples)

energy_consumption = base_load + time_of_day_pattern + temperature_sensitivity + random_variation
energy_consumption = np.maximum(energy_consumption, 10)  # Ensure positive values

# Create DataFrame
df = pd.DataFrame({
    'timestamp': dates,
    'temperature': temperature,
    'humidity': humidity,
    'energy_consumption': energy_consumption
})

# Feature Engineering
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Normalize features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features_to_scale = ['temperature', 'humidity', 'hour', 'day_of_week', 'month']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Save the data
df.to_csv('energy_data.csv', index=False)

print(f"\n✅ Dataset generated with {len(df)} hourly records")
print(f"📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\n📈 Energy Consumption Statistics:")
print(f"   - Average: {df['energy_consumption'].mean():.2f} kWh")
print(f"   - Min: {df['energy_consumption'].min():.2f} kWh")
print(f"   - Max: {df['energy_consumption'].max():.2f} kWh")
print(f"\n🌡️  Temperature Statistics:")
print(f"   - Average: {df['temperature'].mean():.2f}°C")
print(f"   - Min: {df['temperature'].min():.2f}°C")
print(f"   - Max: {df['temperature'].max():.2f}°C")
print("\n✅ Data saved as 'energy_data.csv'")
