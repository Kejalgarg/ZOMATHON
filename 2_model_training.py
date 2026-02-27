import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load enriched data
df = pd.read_csv('data_enriched.csv')

# Remove rows with missing KPT or Time_taken
df = df.dropna(subset=['Kitchen_Prep_Time (min)', 'Time_taken (min)'])

print("=" * 80)
print("KITCHEN PREP TIME (KPT) PREDICTION MODEL")
print("=" * 80)

# Define features and target
categorical_cols = ['City', 'Type_of_restaurant', 'Weatherconditions', 'Road_traffic_density']
numerical_cols = ['Restaurant_rating', 'Restaurant_load', 'Number_of_delivery_partners_nearby', 
                  'Distance_km', 'Dynamic_Surge_Multiplier', 'Dine_in_peak_flag', 
                  'Machine_breakdown_flag', 'Festival', 'multiple_deliveries',
                  'hour_of_day', 'day_of_week', 'is_weekend']

X = df[categorical_cols + numerical_cols]
y = df['Kitchen_Prep_Time (min)']

# Convert categorical to numeric for preprocessing
X_processed = X.copy()
for col in categorical_cols:
    X_processed[col] = pd.factorize(X_processed[col])[0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Preprocessing pipeline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15),
    'XGBoost': XGBRegressor(n_estimators=50, random_state=42, verbosity=0, max_depth=6)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {'model': model, 'mae': mae, 'rmse': rmse, 'y_pred': y_pred}
    
    print(f"  MAE: {mae:.2f} min")
    print(f"  RMSE: {rmse:.2f} min")

# Quantile Regression (for uncertainty)
print("\n" + "=" * 80)
print("QUANTILE REGRESSION (Uncertainty Estimation)")
print("=" * 80)

quantiles = [0.1, 0.5, 0.9]
models_q = {}

for q in quantiles:
    print(f"Training LightGBM for {int(q*100)}th percentile...")
    model = LGBMRegressor(objective='quantile', alpha=q, n_estimators=50, random_state=42, verbose=-1)
    model.fit(X_train_scaled, y_train)
    models_q[q] = model

y_pred_lower = models_q[0.1].predict(X_test_scaled)
y_pred_median = models_q[0.5].predict(X_test_scaled)
y_pred_upper = models_q[0.9].predict(X_test_scaled)

coverage = np.mean((y_test.values >= y_pred_lower) & (y_test.values <= y_pred_upper))
print(f"80% Prediction Interval Coverage: {coverage:.2%}")

# Feature Importance (XGBoost)
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Top 10)")
print("=" * 80)

xgb_model = results['XGBoost']['model']
all_features = categorical_cols + numerical_cols
importance = pd.Series(xgb_model.feature_importances_, index=all_features).sort_values(ascending=False)
print(importance.head(10))

# Save predictions and models
test_results = pd.DataFrame({
    'actual_kpt': y_test.values,
    'pred_kpt_median': y_pred_median,
    'pred_kpt_lower': y_pred_lower,
    'pred_kpt_upper': y_pred_upper,
    'rf_pred': results['Random Forest']['y_pred']
})
test_results.to_csv('test_predictions.csv', index=False)

print("\n✓ Test predictions saved to test_predictions.csv")
print("✓ Models trained and ready for dispatch simulation")
