import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 80)

# Load and prepare data
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

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
print("Training XGBoost for SHAP explanation...")
model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0, max_depth=6)
model.fit(X_train_scaled, y_train)

print("✓ Model trained")

# Create SHAP explainer
print("\nGenerating SHAP values...")
explainer = shap.Explainer(model)
shap_values = explainer(X_test_scaled[:100])

print("✓ SHAP values computed")

# Feature names
feature_names = categorical_cols + numerical_cols

# Summary plot
print("\nCreating SHAP visualizations...")
plt.figure()
shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
print("✓ Summary plot saved to shap_summary.png")

# Bar plot
plt.figure()
shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_names, 
                 plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
print("✓ Bar plot saved to shap_bar.png")

# Waterfall plot for first sample
plt.figure()
shap.plots.waterfall(shap_values[0])
plt.tight_layout()
plt.savefig('shap_waterfall_sample1.png', dpi=150, bbox_inches='tight')
print("✓ Waterfall plot saved to shap_waterfall_sample1.png")

# Save SHAP values for dashboard
np.save('shap_values.npy', shap_values.values)
np.save('X_test_scaled_sample.npy', X_test_scaled[:100])
pd.DataFrame({'feature': feature_names}).to_csv('feature_names.csv', index=False)

print("\n" + "=" * 80)
print("SHAP EXPLAINABILITY INSIGHTS")
print("=" * 80)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)[:10]
for i, (feat, val) in enumerate(top_features, 1):
    print(f"{i}. {feat}: {val:.4f}")

print("\n✓ SHAP analysis complete!")
