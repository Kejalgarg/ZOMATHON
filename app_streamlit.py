import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Zomato KPT Optimization", layout="wide")

st.title("🍕 Zomato Kitchen Prep Time Optimization")
st.markdown("AI-driven dispatch strategy to minimize rider and food wait times")

# Load data directly
@st.cache_data
def load_data():
    try:
        test_preds = pd.read_csv('test_predictions.csv')
        sim_results = pd.read_csv('dispatch_simulation_results.csv')
        enriched_df = pd.read_csv('data_enriched.csv')
        return test_preds, sim_results, enriched_df
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Please run the pipeline scripts first.")
        st.stop()

test_preds, sim_results, enriched_df = load_data()

# ============================================================================
# KEY METRICS
# ============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_kpt = enriched_df['Kitchen_Prep_Time (min)'].mean()
    st.metric("Avg KPT", f"{avg_kpt:.1f} min")

with col2:
    model_mae = 6.18
    st.metric("Model MAE", f"{model_mae:.2f} min")

with col3:
    baseline_wait = sim_results['rider_wait_baseline'].mean()
    optimized_wait = sim_results['rider_wait_optimized'].mean()
    total_reduction = (baseline_wait + sim_results['food_wait_baseline'].mean()) - \
                     (optimized_wait + sim_results['food_wait_optimized'].mean())
    st.metric("Total Wait Reduction", f"{total_reduction:.2f} min")

with col4:
    coverage = 79.47
    st.metric("Prediction Interval Coverage", f"{coverage:.1f}%")

# ============================================================================
# DISPATCH STRATEGY COMPARISON
# ============================================================================
st.subheader("📊 Dispatch Strategy Comparison")

col1, col2 = st.columns(2)

with col1:
    strategies = ['Baseline\n(Immediate)', 'Optimized\n(Median)', 'Pessimistic\n(90th %ile)']
    rider_waits = [
        sim_results['rider_wait_baseline'].mean(),
        sim_results['rider_wait_optimized'].mean(),
        sim_results['rider_wait_pessimistic'].mean()
    ]
    
    fig_rider = go.Figure(data=[
        go.Bar(x=strategies, y=rider_waits, marker_color=['#ef553b', '#00cc96', '#636efa'],
               text=[f'{w:.2f} min' for w in rider_waits], textposition='outside')
    ])
    fig_rider.update_layout(title="Average Rider Wait Time", yaxis_title="Minutes", 
                           xaxis_title="Strategy", height=400, showlegend=False)
    st.plotly_chart(fig_rider, width='stretch')

with col2:
    food_waits = [
        sim_results['food_wait_baseline'].mean(),
        sim_results['food_wait_optimized'].mean(),
        sim_results['food_wait_pessimistic'].mean()
    ]
    
    fig_food = go.Figure(data=[
        go.Bar(x=strategies, y=food_waits, marker_color=['#ef553b', '#00cc96', '#636efa'],
               text=[f'{w:.2f} min' for w in food_waits], textposition='outside')
    ])
    fig_food.update_layout(title="Average Food Wait Time", yaxis_title="Minutes", 
                          xaxis_title="Strategy", height=400, showlegend=False)
    st.plotly_chart(fig_food, width='stretch')

# ============================================================================
# TOTAL WAIT TIME (COMBINED)
# ============================================================================
st.subheader("⏱️ Total Wait Time (Rider + Food)")

total_waits = [
    sim_results['rider_wait_baseline'].mean() + sim_results['food_wait_baseline'].mean(),
    sim_results['rider_wait_optimized'].mean() + sim_results['food_wait_optimized'].mean(),
    sim_results['rider_wait_pessimistic'].mean() + sim_results['food_wait_pessimistic'].mean()
]

fig_total = go.Figure(data=[
    go.Bar(x=strategies, y=total_waits, marker_color=['#ef553b', '#00cc96', '#636efa'],
           text=[f'{w:.2f} min' for w in total_waits], textposition='outside')
])
fig_total.update_layout(title="Total System Wait Time (Lower is Better)", 
                       yaxis_title="Minutes", xaxis_title="Strategy", 
                       height=400, showlegend=False)
st.plotly_chart(fig_total, width='stretch')

# ============================================================================
# PREDICTION ACCURACY
# ============================================================================
st.subheader("🎯 Model Predictions vs Actuals")

col1, col2 = st.columns(2)

with col1:
    sample_size = min(500, len(test_preds))
    test_sample = test_preds.head(sample_size)
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=test_sample['actual_kpt'], y=test_sample['pred_kpt_median'],
                                    mode='markers', name='Predictions',
                                    marker=dict(size=6, opacity=0.6, color='#636efa')))
    fig_scatter.add_trace(go.Scatter(x=[test_sample['actual_kpt'].min(), test_sample['actual_kpt'].max()],
                                    y=[test_sample['actual_kpt'].min(), test_sample['actual_kpt'].max()],
                                    mode='lines', name='Perfect Prediction', 
                                    line=dict(dash='dash', color='red')))
    fig_scatter.update_layout(title='Actual vs Predicted KPT', height=400,
                             xaxis_title='Actual KPT (min)', yaxis_title='Predicted KPT (min)')
    st.plotly_chart(fig_scatter, width='stretch')

with col2:
    test_sample_copy = test_sample.copy()
    test_sample_copy['interval_width'] = test_sample_copy['pred_kpt_upper'] - test_sample_copy['pred_kpt_lower']
    
    fig_interval = go.Figure(data=[
        go.Histogram(x=test_sample_copy['interval_width'], nbinsx=30, 
                    marker=dict(color='#00cc96'))
    ])
    fig_interval.update_layout(title='Distribution of 80% Prediction Intervals',
                              xaxis_title='Prediction Interval Width (min)',
                              yaxis_title='Frequency', height=400, showlegend=False)
    st.plotly_chart(fig_interval, width='stretch')

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
st.subheader("🔍 Top Factors Influencing Kitchen Prep Time")

features = ['Machine Breakdown', 'Dine-in Peak', 'Restaurant Type', 'Restaurant Load', 
            'Restaurant Rating', 'Festival', 'Multiple Deliveries', 'Distance', 
            'Delivery Partners', 'Surge Multiplier']
importances = [0.298, 0.290, 0.273, 0.041, 0.035, 0.014, 0.006, 0.006, 0.006, 0.006]

fig_importance = go.Figure(data=[
    go.Bar(y=features, x=importances, orientation='h',
           marker=dict(color=importances, colorscale='Viridis', showscale=True),
           text=[f'{x:.3f}' for x in importances], textposition='outside')
])
fig_importance.update_layout(title='Feature Importance Scores (XGBoost)',
                            xaxis_title='Importance Score', height=400, showlegend=False)
st.plotly_chart(fig_importance, width='stretch')

# ============================================================================
# KEY INSIGHTS
# ============================================================================
st.subheader("💡 Key Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🎯 Optimized Strategy Benefits:**
    - **71.7% reduction** in total wait time (21.69 → 6.13 min)
    - Rider wait reduced from 0.19 to 3.06 min (acceptable delay)
    - Food wait reduced from 21.50 to 3.08 min (major improvement)
    - Better customer experience with hot food delivery
    """)

with col2:
    st.markdown("""
    **⚠️ Pessimistic Strategy (Conservative):**
    - Nearly **eliminates food waste** (0.23 min avg)
    - Rider wait increases to 9.64 min (trade-off)
    - Suitable when food quality is critical
    - **80% prediction interval coverage** validates model reliability
    """)

st.markdown("""
---
### 📈 Implementation Recommendation
Use the **Optimized Strategy** (median predictions) as default:
1. Provides ~72% reduction in total wait time
2. Balances rider and food wait efficiently
3. Backed by 79.47% prediction interval coverage
4. Machine breakdown flag is the strongest KPT predictor – monitor closely
""")
