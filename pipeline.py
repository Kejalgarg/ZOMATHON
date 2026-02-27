import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Zomato_Enterprise_Level_Dataset_50k.csv')

# Extract hour FIRST (before any time conversions)
df['hour_of_day'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S').dt.hour

# Then do other conversions
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['day_of_week'] = df['Order_Date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Remove rows with missing critical columns
df = df.dropna(subset=['Kitchen_Prep_Time (min)', 'Time_taken (min)'])

print("✓ Data loaded and features engineered")
print(f"  Dataset shape: {df.shape}")

# Define features and target
categorical_cols = ['City', 'Type_of_restaurant', 'Weatherconditions', 'Road_traffic_density']
numerical_cols = ['Restaurant_rating', 'Restaurant_load', 'Number_of_delivery_partners_nearby', 
                  'Distance_km', 'Dynamic_Surge_Multiplier', 'Dine_in_peak_flag', 
                  'Machine_breakdown_flag', 'Festival', 'multiple_deliveries',
                  'hour_of_day', 'day_of_week', 'is_weekend']

X = df[categorical_cols + numerical_cols].copy()

# Convert categorical to numeric
for col in categorical_cols:
    X[col] = pd.factorize(X[col])[0]

y = df['Kitchen_Prep_Time (min)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0, max_depth=6)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"✓ Model trained - MAE: {mae:.2f} min, RMSE: {rmse:.2f} min")

# Save enriched data
df.to_csv('data_enriched.csv', index=False)
print("✓ Pipeline complete - data_enriched.csv saved")
