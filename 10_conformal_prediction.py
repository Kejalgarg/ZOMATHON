import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

print("=" * 80)
print("CONFORMAL PREDICTION FOR RIGOROUS UNCERTAINTY")
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

# Split: train, calibration, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nData Split:")
print(f"  Train: {len(X_train)} samples")
print(f"  Calibration: {len(X_cal)} samples (for conformal calibration)")
print(f"  Test: {len(X_test)} samples")

# Train base model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0, max_depth=6)
model.fit(X_train_scaled, y_train)

# Get predictions
y_train_pred = model.predict(X_train_scaled)
y_cal_pred = model.predict(X_cal_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate residuals on calibration set
residuals_cal = np.abs(y_cal - y_cal_pred)

# Conformal Prediction at different confidence levels
def conformal_intervals(y_pred, residuals_cal, alpha=0.1):
    """
    Create prediction intervals with (1-alpha) coverage guarantee
    alpha=0.1 → 90% coverage (10% error rate)
    """
    # Use quantile of calibration residuals
    quantile_val = np.quantile(residuals_cal, 1 - alpha)
    lower = y_pred - quantile_val
    upper = y_pred + quantile_val
    return lower, upper

# Test different confidence levels
confidence_levels = [0.80, 0.85, 0.90, 0.95]
results = []

print("\n" + "=" * 80)
print("CONFORMAL PREDICTION RESULTS")
print("=" * 80)

for conf in confidence_levels:
    alpha = 1 - conf
    lower, upper = conformal_intervals(y_test_pred, residuals_cal, alpha=alpha)
    
    # Calculate coverage (what % of true values fall in interval?)
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    interval_width = np.mean(upper - lower)
    
    results.append({
        'Confidence': f'{conf*100:.0f}%',
        'Alpha': alpha,
        'Coverage': f'{coverage*100:.1f}%',
        'Avg Interval Width': f'{interval_width:.2f} min',
        'Lower Quantile': np.quantile(residuals_cal, 1 - alpha)
    })
    
    print(f"\n{conf*100:.0f}% Confidence Level (α={alpha}):")
    print(f"  ✓ Theoretical Coverage: {conf*100:.0f}%")
    print(f"  ✓ Actual Coverage: {coverage*100:.1f}% (on test set)")
    print(f"  ✓ Avg Prediction Interval Width: {interval_width:.2f} minutes")
    print(f"    (e.g., prediction: 28.5 ± {interval_width/2:.1f} min)")

# Example predictions with 90% Confidence
print("\n" + "=" * 80)
print("EXAMPLE: 90% CONFIDENCE INTERVALS")
print("=" * 80)

alpha_90 = 0.10
lower_90, upper_90 = conformal_intervals(y_test_pred, residuals_cal, alpha=alpha_90)

# Show first 10 examples
print("\nFirst 10 Test Predictions:")
print(f"{'True KPT':>10} {'Predicted':>10} {'90% Lower':>12} {'90% Upper':>12} {'Width':>10} {'Accurate':>10}")
print("-" * 65)

accurate_count = 0
for i in range(min(10, len(y_test))):
    accurate = "✓" if (y_test.iloc[i] >= lower_90[i] and y_test.iloc[i] <= upper_90[i]) else "✗"
    if accurate == "✓":
        accurate_count += 1
    width = upper_90[i] - lower_90[i]
    print(f"{y_test.iloc[i]:10.1f} {y_test_pred[i]:10.1f} {lower_90[i]:12.1f} {upper_90[i]:12.1f} {width:10.1f} {accurate:>10}")

# Visualization
fig = go.Figure()

# Add scatter plot
fig.add_trace(go.Scatter(
    x=np.arange(len(y_test)),
    y=y_test.values,
    mode='markers',
    name='True KPT',
    marker=dict(size=8, color='#636efa')
))

fig.add_trace(go.Scatter(
    x=np.arange(len(y_test)),
    y=y_test_pred,
    mode='markers',
    name='Predicted KPT',
    marker=dict(size=6, color='#ef553b', symbol='x')
))

# Add 90% confidence interval as ribbon
fig.add_trace(go.Scatter(
    x=np.arange(len(y_test)),
    y=upper_90,
    fill=None,
    mode='lines',
    name='90% Upper Bound',
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=np.arange(len(y_test)),
    y=lower_90,
    fill='tonexty',
    mode='lines',
    name='90% Confidence Interval',
    line=dict(width=0),
    fillcolor='rgba(0, 204, 150, 0.2)'
))

fig.update_layout(
    title='Conformal Prediction: 90% Coverage Intervals',
    xaxis_title='Test Sample Index',
    yaxis_title='Kitchen Prep Time (min)',
    height=500,
    hovermode='x unified'
)
fig.write_html('conformal_prediction.html')
print("\n✓ Conformal prediction plot saved to conformal_prediction.html")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('conformal_results.csv', index=False)
print("\n✓ Conformal results saved to conformal_results.csv")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("""
Conformal Prediction provides:
✓ Distribution-free coverage guarantees (no assumptions about data distribution)
✓ Finite-sample rigor (actually works, not asymptotic)
✓ Simple to understand: "This prediction is 28.5 ± 4.2 minutes with 90% confidence"

Use Case for Zomato:
- Show customers: "Your food arrives 28-37 minutes (90% confidence)"
- Dispatch riders based on interval width (narrow = high confidence, dispatch early)
- Adjust delivery fees if interval is wide (need buffer)
""")
