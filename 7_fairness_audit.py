import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FAIRNESS AUDIT")
print("=" * 80)

# Load data
df = pd.read_csv('data_enriched.csv')
df = df.dropna(subset=['Kitchen_Prep_Time (min)'])

categorical_cols = ['City', 'Type_of_restaurant', 'Weatherconditions', 'Road_traffic_density']
numerical_cols = ['Restaurant_rating', 'Restaurant_load', 'Number_of_delivery_partners_nearby', 
                  'Distance_km', 'Dynamic_Surge_Multiplier', 'Dine_in_peak_flag', 
                  'Machine_breakdown_flag', 'Festival', 'multiple_deliveries',
                  'hour_of_day', 'day_of_week', 'is_weekend']

X = df[categorical_cols + numerical_cols].copy()
for col in categorical_cols:
    X[col] = pd.factorize(X[col])[0]

y = df['Kitchen_Prep_Time (min)']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add sensitive features back
test_df = df.iloc[X_test.index].copy()
test_df['y_true'] = y_test.values

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0, max_depth=6)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
test_df['y_pred'] = y_pred

# Fairness analysis by sensitive features
print("\n📊 MODEL PERFORMANCE BY RESTAURANT TYPE:")
print("=" * 80)
for rest_type in test_df['Type_of_restaurant'].unique():
    subset = test_df[test_df['Type_of_restaurant'] == rest_type]
    if len(subset) > 0:
        mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
        rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
        print(f"  {rest_type}: MAE={mae:.2f} min, RMSE={rmse:.2f} min, Samples={len(subset)}")

print("\n📊 MODEL PERFORMANCE BY CITY:")
print("=" * 80)
for city in test_df['City'].unique()[:5]:
    subset = test_df[test_df['City'] == city]
    if len(subset) > 0:
        mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
        rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
        print(f"  {city}: MAE={mae:.2f} min, RMSE={rmse:.2f} min, Samples={len(subset)}")

print("\n📊 MODEL PERFORMANCE BY WEATHER:")
print("=" * 80)
for weather in test_df['Weatherconditions'].unique():
    subset = test_df[test_df['Weatherconditions'] == weather]
    if len(subset) > 0:
        mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
        rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
        print(f"  {weather}: MAE={mae:.2f} min, RMSE={rmse:.2f} min, Samples={len(subset)}")

print("\n📊 MODEL PERFORMANCE BY PEAK FLAG:")
print("=" * 80)
for peak in [0, 1]:
    subset = test_df[test_df['Dine_in_peak_flag'] == peak]
    if len(subset) > 0:
        peak_label = "Peak Hours" if peak == 1 else "Off-Peak"
        mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
        rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
        print(f"  {peak_label}: MAE={mae:.2f} min, RMSE={rmse:.2f} min, Samples={len(subset)}")

# Overall fairness score
print("\n" + "=" * 80)
print("FAIRNESS DISPARITY ANALYSIS")
print("=" * 80)

restaurant_maes = []
for rest_type in test_df['Type_of_restaurant'].unique():
    subset = test_df[test_df['Type_of_restaurant'] == rest_type]
    mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
    restaurant_maes.append(mae)

disparity = max(restaurant_maes) / min(restaurant_maes) if len(restaurant_maes) > 0 else 1.0
print(f"Restaurant Type Disparity Ratio: {disparity:.2f}x")

if disparity < 1.2:
    print("✓ EXCELLENT: Model is fair across restaurant types!")
elif disparity < 1.5:
    print("✓ GOOD: Model is reasonably fair")
else:
    print("⚠ WARNING: Model shows bias - may need retraining with fairness constraints")

# Save fairness report
overall_mae = mean_absolute_error(test_df['y_true'], test_df['y_pred'])
overall_rmse = np.sqrt(mean_squared_error(test_df['y_true'], test_df['y_pred']))

fairness_report = pd.DataFrame({
    'overall_mae': [overall_mae],
    'overall_rmse': [overall_rmse],
    'disparity_ratio': [disparity],
    'timestamp': [pd.Timestamp.now()]
})
fairness_report.to_csv('fairness_report.csv', index=False)
print("\n✓ Fairness report saved to fairness_report.csv")
