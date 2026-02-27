import pandas as pd
import numpy as np

print("=" * 80)
print("DISPATCH STRATEGY SIMULATION")
print("=" * 80)

# Load data with predictions
test_df = pd.read_csv('test_predictions.csv')
full_df = pd.read_csv('data_with_travel_time.csv')

# Use first 1000 predictions for simulation
test_df = test_df.head(1000)

# Merge with first-mile data
first_mile_sample = full_df[['first_mile_time']].head(1000).reset_index(drop=True)
test_df = pd.concat([test_df.reset_index(drop=True), first_mile_sample], axis=1)

results = []

for idx, row in test_df.iterrows():
    actual_kpt = row['actual_kpt']
    pred_kpt_median = row['pred_kpt_median']
    pred_kpt_upper = row['pred_kpt_upper']
    first_mile = row['first_mile_time']
    
    # BASELINE: Dispatch immediately
    arrival_baseline = first_mile
    rider_wait_baseline = max(0, arrival_baseline - actual_kpt)
    food_wait_baseline = max(0, actual_kpt - arrival_baseline)
    
    # OPTIMIZED (Median): Delay dispatch to match median prediction
    delay_median = max(0, pred_kpt_median - first_mile)
    arrival_optimized = delay_median + first_mile
    rider_wait_optimized = max(0, arrival_optimized - actual_kpt)
    food_wait_optimized = max(0, actual_kpt - arrival_optimized)
    
    # PESSIMISTIC (90th percentile): Very conservative
    delay_pessimistic = max(0, pred_kpt_upper - first_mile)
    arrival_pessimistic = delay_pessimistic + first_mile
    rider_wait_pessimistic = max(0, arrival_pessimistic - actual_kpt)
    food_wait_pessimistic = max(0, actual_kpt - arrival_pessimistic)
    
    results.append({
        'actual_kpt': actual_kpt,
        'pred_kpt_median': pred_kpt_median,
        'first_mile': first_mile,
        'rider_wait_baseline': rider_wait_baseline,
        'food_wait_baseline': food_wait_baseline,
        'rider_wait_optimized': rider_wait_optimized,
        'food_wait_optimized': food_wait_optimized,
        'rider_wait_pessimistic': rider_wait_pessimistic,
        'food_wait_pessimistic': food_wait_pessimistic,
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("BASELINE STRATEGY (Dispatch Immediately)")
print("=" * 80)
print(f"  Average Rider Wait: {results_df['rider_wait_baseline'].mean():.2f} min")
print(f"  Average Food Wait: {results_df['food_wait_baseline'].mean():.2f} min")
print(f"  Total Wait (Rider + Food): {(results_df['rider_wait_baseline'] + results_df['food_wait_baseline']).mean():.2f} min")

print("\n" + "=" * 80)
print("OPTIMIZED STRATEGY (Delay using Median KPT Prediction)")
print("=" * 80)
print(f"  Average Rider Wait: {results_df['rider_wait_optimized'].mean():.2f} min")
print(f"  Average Food Wait: {results_df['food_wait_optimized'].mean():.2f} min")
print(f"  Total Wait (Rider + Food): {(results_df['rider_wait_optimized'] + results_df['food_wait_optimized']).mean():.2f} min")

rider_wait_reduction = (results_df['rider_wait_baseline'].mean() - results_df['rider_wait_optimized'].mean()) / results_df['rider_wait_baseline'].mean() * 100 if results_df['rider_wait_baseline'].mean() > 0 else 0
print(f"\n  ✓ Rider Wait Reduction: {rider_wait_reduction:.1f}%")

print("\n" + "=" * 80)
print("PESSIMISTIC STRATEGY (Delay using 90th Percentile KPT Prediction)")
print("=" * 80)
print(f"  Average Rider Wait: {results_df['rider_wait_pessimistic'].mean():.2f} min")
print(f"  Average Food Wait: {results_df['food_wait_pessimistic'].mean():.2f} min")
print(f"  Total Wait (Rider + Food): {(results_df['rider_wait_pessimistic'] + results_df['food_wait_pessimistic']).mean():.2f} min")

rider_wait_reduction_pess = (results_df['rider_wait_baseline'].mean() - results_df['rider_wait_pessimistic'].mean()) / results_df['rider_wait_baseline'].mean() * 100 if results_df['rider_wait_baseline'].mean() > 0 else 0
print(f"\n  ✓ Rider Wait Reduction (vs Baseline): {rider_wait_reduction_pess:.1f}%")

results_df.to_csv('dispatch_simulation_results.csv', index=False)
print("\n✓ Simulation results saved to dispatch_simulation_results.csv")
