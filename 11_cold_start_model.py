import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

print("=" * 80)
print("COLD-START MODEL: ESTIMATE KPT FOR NEW RESTAURANTS")
print("=" * 80)

# Load data
df = pd.read_csv('data_enriched.csv')
df = df.dropna(subset=['Kitchen_Prep_Time (min)'])

# Group by restaurant and get aggregate stats
restaurant_features = df.groupby('City').agg({
    'Kitchen_Prep_Time (min)': ['mean', 'std', 'min', 'max'],
    'Restaurant_rating': 'mean',
    'Type_of_restaurant': lambda x: pd.Series.mode(x)[0] if len(pd.Series.mode(x)) > 0 else 'Cloud Kitchen',
    'Weatherconditions': 'count'  # proxy for activity level
}).reset_index()

restaurant_features.columns = ['City', 'avg_kpt', 'std_kpt', 'min_kpt', 'max_kpt', 
                                'avg_rating', 'dominant_type', 'order_volume']

print(f"\nNewly discovered restaurant profiles by city:")
print(restaurant_features.to_string(index=False))

# Create cold-start model features (use aggregate statistics)
X_cold = pd.DataFrame()
X_cold['avg_city_kpt'] = restaurant_features['avg_kpt']
X_cold['std_city_kpt'] = restaurant_features['std_kpt']
X_cold['avg_rating'] = restaurant_features['avg_rating']
X_cold['order_volume'] = restaurant_features['order_volume']

# Add restaurant type encoding
type_map = {'Home Kitchen': 0, 'Cloud Kitchen': 1, 'Dine-in': 2}
X_cold['restaurant_type'] = restaurant_features['dominant_type'].map(type_map)

# Target: average KPT for each city
y_cold = restaurant_features['avg_kpt']

# Train cold-start model
scaler = StandardScaler()
X_cold_scaled = scaler.fit_transform(X_cold)

cold_start_model = GradientBoostingRegressor(n_estimators=30, random_state=42, max_depth=4)
cold_start_model.fit(X_cold_scaled, y_cold)

print("\n" + "=" * 80)
print("COLD-START MODEL PREDICTIONS FOR NEW RESTAURANTS")
print("=" * 80)

# Simulate new restaurant arrivals
new_restaurants = pd.DataFrame({
    'City': ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'],
    'Type': ['Cloud Kitchen', 'Home Kitchen', 'Dine-in', 'Cloud Kitchen', 'Home Kitchen'],
    'Rating': [4.2, 3.8, 4.5, 4.0, 3.9]
})

print("\nNew Restaurants (No Historical Data):")
print(new_restaurants.to_string(index=False))

# Make predictions
predictions = []
for _, new_rest in new_restaurants.iterrows():
    city = new_rest['City']
    
    # Find city profile
    city_profile = restaurant_features[restaurant_features['City'] == city]
    
    if len(city_profile) > 0:
        avg_city_kpt = city_profile['avg_kpt'].values[0]
        std_city_kpt = city_profile['std_kpt'].values[0]
        avg_rating = new_rest['Rating']
        order_volume = city_profile['order_volume'].values[0]
        restaurant_type = type_map.get(new_rest['Type'], 1)
        
        # Create feature vector
        X_new = np.array([[avg_city_kpt, std_city_kpt, avg_rating, order_volume, restaurant_type]])
        X_new_scaled = scaler.transform(X_new)
        
        # Predict
        pred_kpt = cold_start_model.predict(X_new_scaled)[0]
        
        predictions.append({
            'Restaurant': f"{new_rest['City']} - {new_rest['Type']}",
            'Estimated KPT': round(pred_kpt, 1),
            'Confidence': 'Medium (based on city avg)',
            'Data Points': 'None (cold-start)',
            'Recommendation': 'Switch to full model after 50 orders'
        })
    
    else:
        # Default to global mean
        global_avg = df['Kitchen_Prep_Time (min)'].mean()
        predictions.append({
            'Restaurant': f"{new_rest['City']} - {new_rest['Type']}",
            'Estimated KPT': round(global_avg, 1),
            'Confidence': 'Low (global default)',
            'Data Points': 'None (city unknown)',
            'Recommendation': 'Collect 10 orders urgently'
        })

preds_df = pd.DataFrame(predictions)
print("\n" + "=" * 80)
print("PREDICTIONS")
print("=" * 80)
print(preds_df.to_string(index=False))

# Visualization: Comparison
restaurants_list = [p['Restaurant'] for p in predictions]
estimated_kpts = [p['Estimated KPT'] for p in predictions]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=restaurants_list,
    y=estimated_kpts,
    name='Cold-Start Estimate',
    marker_color='#636efa'
))

fig.add_hline(y=df['Kitchen_Prep_Time (min)'].mean(), 
             line_dash="dash", line_color="red",
             annotation_text="Global Average",
             annotation_position="right")

fig.update_layout(
    title='Cold-Start KPT Estimates for New Restaurants',
    yaxis_title='Estimated KPT (minutes)',
    xaxis_title='Restaurant',
    height=400
)
fig.write_html('cold_start_predictions.html')
print("\n✓ Cold-start predictions plot saved to cold_start_predictions.html")

# Save model strategy
strategy = """
COLD-START STRATEGY:

Phase 1 (Orders 0-10):
  - Use cold-start model (geography-based)
  - E.g., "Mumbai home kitchens avg 28.5 min"

Phase 2 (Orders 10-50):
  - Blend cold-start + restaurant's own data
  - Gradually increase restaurant data weight

Phase 3 (Orders 50+):
  - Switch to full XGBoost model
  - Use restaurant's historical patterns

Benefits:
✓ Day 1 operational (no cold-start problem)
✓ Smooth transition to personalized model
✓ Fairness to new partners
"""

print("\n" + "=" * 80)
print(strategy)
print("=" * 80)

# Save
preds_df.to_csv('cold_start_predictions.csv', index=False)
print("\n✓ Saved to cold_start_predictions.csv")
