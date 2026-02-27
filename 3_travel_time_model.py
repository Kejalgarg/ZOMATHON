import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_enriched.csv')
df = df.dropna(subset=['Kitchen_Prep_Time (min)', 'Time_taken (min)'])

print("=" * 80)
print("TRAVEL TIME MODEL")
print("=" * 80)

# Compute travel time (total - prep time)
df['travel_time_total'] = df['Time_taken (min)'] - df['Kitchen_Prep_Time (min)']
df = df[df['travel_time_total'] >= 0]

print(f"Average travel time: {df['travel_time_total'].mean():.2f} min")
print(f"Travel time range: {df['travel_time_total'].min():.2f} - {df['travel_time_total'].max():.2f} min")

# Features for travel time prediction
categorical_cols = ['City', 'Weatherconditions', 'Road_traffic_density']
numerical_cols = ['Distance_km', 'Dynamic_Surge_Multiplier', 'hour_of_day']

X_tt = df[categorical_cols + numerical_cols].copy()

# Convert categorical to numeric
for col in categorical_cols:
    X_tt[col] = pd.factorize(X_tt[col])[0]

y_tt = df['travel_time_total']

X_train_tt, X_test_tt, y_train_tt, y_test_tt = train_test_split(X_tt, y_tt, test_size=0.2, random_state=42)

scaler_tt = StandardScaler()
X_train_tt_scaled = scaler_tt.fit_transform(X_train_tt)
X_test_tt_scaled = scaler_tt.transform(X_test_tt)

model_tt = XGBRegressor(n_estimators=50, random_state=42, verbosity=0, max_depth=6)
model_tt.fit(X_train_tt_scaled, y_train_tt)

y_pred_tt = model_tt.predict(X_test_tt_scaled)
mae_tt = mean_absolute_error(y_test_tt, y_pred_tt)
rmse_tt = np.sqrt(mean_squared_error(y_test_tt, y_pred_tt))

print(f"Travel Time Model MAE: {mae_tt:.2f} min")
print(f"Travel Time Model RMSE: {rmse_tt:.2f} min")

# Simulate first-mile and last-mile split
df['first_mile_frac'] = 0.4
df['first_mile_time'] = df['travel_time_total'] * df['first_mile_frac']
df['last_mile_time'] = df['travel_time_total'] * (1 - df['first_mile_frac'])

print(f"\nFirst-Mile Average: {df['first_mile_time'].mean():.2f} min")
print(f"Last-Mile Average: {df['last_mile_time'].mean():.2f} min")

df.to_csv('data_with_travel_time.csv', index=False)
print("\n✓ Travel time features added and saved to data_with_travel_time.csv")
