import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="What-If Analysis", layout="wide")

st.title("🎯 What-If Analysis: KPT Prediction Tool")
st.markdown("Adjust features to see impact on Kitchen Prep Time and optimal dispatch strategy")

@st.cache_resource
def load_model():
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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0, max_depth=6)
    model.fit(X_scaled, y)
    
    return model, scaler, categorical_cols, numerical_cols, df

model, scaler, categorical_cols, numerical_cols, df = load_model()

st.sidebar.header("⚙️ Adjust Features")

restaurant_load = st.sidebar.slider("Restaurant Load (pending orders)", 0, 100, 50)
machine_breakdown = st.sidebar.checkbox("Machine Breakdown?", value=False)
restaurant_rating = st.sidebar.slider("Restaurant Rating", 2.5, 5.0, 4.1, 0.1)

peak_hour = st.sidebar.checkbox("Peak Hour?", value=False)
festival = st.sidebar.checkbox("Festival/Event?", value=False)
distance = st.sidebar.slider("Distance (km)", 1.0, 20.0, 10.0, 0.5)

surge_multiplier = st.sidebar.slider("Surge Multiplier", 0.5, 3.0, 1.0, 0.1)
delivery_partners = st.sidebar.slider("Delivery Partners Nearby", 1, 30, 10)

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Day of Week", 
                           ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                            'Friday', 'Saturday', 'Sunday'])

day_map = {day: i for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                           'Friday', 'Saturday', 'Sunday'])}
day_num = day_map[day_of_week]
is_weekend = 1 if day_num >= 5 else 0

input_dict = {
    'Restaurant_rating': restaurant_rating,
    'Restaurant_load': restaurant_load,
    'Number_of_delivery_partners_nearby': delivery_partners,
    'Distance_km': distance,
    'Dynamic_Surge_Multiplier': surge_multiplier,
    'Dine_in_peak_flag': int(peak_hour),
    'Machine_breakdown_flag': int(machine_breakdown),
    'Festival': int(festival),
    'multiple_deliveries': 0,
    'hour_of_day': hour,
    'day_of_week': day_num,
    'is_weekend': is_weekend,
    'City': 0,
    'Type_of_restaurant': 0,
    'Weatherconditions': 0,
    'Road_traffic_density': 0,
}

X_input = pd.DataFrame([input_dict])[categorical_cols + numerical_cols]
X_input_scaled = scaler.transform(X_input)

pred_kpt = model.predict(X_input_scaled)[0]

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Predicted KPT", f"{pred_kpt:.1f} min")

with col2:
    if pred_kpt < 20:
        strategy = "🚀 Immediate Dispatch"
    elif pred_kpt < 35:
        strategy = "⚡ Optimized (Median)"
    else:
        strategy = "🛡️ Pessimistic (90th %ile)"
    st.metric("Strategy", strategy)

with col3:
    estimated_wait = pred_kpt * 0.2
    st.metric("Est. Rider Wait", f"{estimated_wait:.1f} min")

st.subheader("📋 Dispatch Strategy Comparison")

strategies_data = {
    'Strategy': ['Immediate', 'Optimized', 'Pessimistic'],
    'Rider Wait': [max(0, 15 - pred_kpt), max(0, 3), 0],
    'Food Wait': [0, max(0, 3), max(0, 15)]
}

fig = go.Figure()
fig.add_trace(go.Bar(x=strategies_data['Strategy'], y=strategies_data['Rider Wait'],
                     name='Rider Wait', marker_color='#ef553b'))
fig.add_trace(go.Bar(x=strategies_data['Strategy'], y=strategies_data['Food Wait'],
                     name='Food Wait', marker_color='#00cc96'))

fig.update_layout(barmode='stack', title='Wait Time by Strategy',
                 xaxis_title='Strategy', yaxis_title='Minutes', height=400)
st.plotly_chart(fig, width='stretch')

st.subheader("💡 AI Recommendations")

recommendations = []

if machine_breakdown:
    recommendations.append("🔧 **Machine Breakdown Detected**: Prioritize maintenance!")

if peak_hour and restaurant_load > 70:
    recommendations.append("⚠️ **High Load at Peak**: Consider temporary staff or longer prep buffer")

if distance > 15:
    recommendations.append("📍 **Far Distance**: Use pessimistic dispatch to avoid rider wait")

if surge_multiplier > 2:
    recommendations.append("💰 **High Surge**: Orders are delayed - consider dynamic pricing adjustment")

if pred_kpt > 40:
    recommendations.append("⏱️ **Long Prep Time**: Material for operational improvement")
else:
    recommendations.append("✓ **Normal Prep Time**: All systems functioning well")

for rec in recommendations:
    st.info(rec)
