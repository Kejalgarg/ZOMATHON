import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Zomato_Enterprise_Level_Dataset_50k.csv')

print("=" * 80)
print("DATASET INFO")
print("=" * 80)
print(df.info())

print("\n" + "=" * 80)
print("BASIC STATISTICS")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
print(df.isnull().sum())

# Convert date/time columns
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S').dt.time
df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], format='%H:%M:%S').dt.time

# Feature Engineering
df['hour_of_day'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S').dt.hour
df['day_of_week'] = df['Order_Date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING - NEW COLUMNS CREATED")
print("=" * 80)
print(df[['Order_Date', 'hour_of_day', 'day_of_week', 'is_weekend']].head(10))

# Save enriched dataset
df.to_csv('data_enriched.csv', index=False)
print("\n✓ Enriched dataset saved to data_enriched.csv")
