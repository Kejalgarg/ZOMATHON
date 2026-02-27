import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LSTM MODEL FOR TEMPORAL KPT PREDICTION")
print("=" * 80)

# Load data
df = pd.read_csv('data_enriched.csv')
df = df.dropna(subset=['Kitchen_Prep_Time (min)', 'Time_taken (min)'])
df = df.sort_values('Order_Date').reset_index(drop=True)

# Create sequences per city (proxy for restaurant patterns)
def create_sequences(group, seq_length=10):
    """Create sequences of past 10 orders"""
    features_cols = ['Restaurant_rating', 'Restaurant_load', 'hour_of_day', 
                     'Dynamic_Surge_Multiplier', 'is_weekend']
    
    xs, ys = [], []
    features = group[features_cols].values
    target = group['Kitchen_Prep_Time (min)'].values
    
    for i in range(len(group) - seq_length):
        xs.append(features[i:i+seq_length])
        ys.append(target[i+seq_length])
    
    return np.array(xs), np.array(ys)

X_seqs, y_seqs = [], []
for city in df['City'].unique()[:5]:
    city_data = df[df['City'] == city]
    if len(city_data) > 15:
        X, y = create_sequences(city_data, seq_length=10)
        if len(X) > 0:
            X_seqs.append(X)
            y_seqs.append(y)

if X_seqs:
    X = np.vstack(X_seqs)
    y = np.hstack(y_seqs)
    
    print(f"Sequences created: {X.shape[0]} samples")
    print(f"Sequence shape: {X.shape[1]} timesteps, {X.shape[2]} features")
    
    # Scale features
    n_samples, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(n_samples, seq_len, n_features)
    
    # Train/val split
    split = int(0.8 * len(X))
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\nTrain set: {X_train.shape[0]} sequences")
    print(f"Validation set: {X_val.shape[0]} sequences")
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    
    print("\n" + "=" * 80)
    print("TRAINING LSTM")
    print("=" * 80)
    
    early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(X_train, y_train, 
                       validation_data=(X_val, y_val),
                       epochs=20, batch_size=32, 
                       callbacks=[early_stop], verbose=1)
    
    # Evaluate
    val_mae = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"\n✓ LSTM Model trained")
    print(f"  Validation MAE: {val_mae:.2f} min")
    
    # Save model
    model.save('lstm_kpt_model.h5')
    np.save('lstm_scaler_params.npy', [scaler.mean_, scaler.scale_])
    print("✓ Model saved to lstm_kpt_model.h5")
else:
    print("⚠ Not enough sequences for LSTM training")
