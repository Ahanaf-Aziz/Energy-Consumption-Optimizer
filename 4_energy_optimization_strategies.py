import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 60)
print("STEP 3: TRAINING LSTM DEEP LEARNING MODEL")
print("=" * 60)

# Load data
df = pd.read_csv('energy_data.csv')

# Prepare sequences for LSTM
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Use energy consumption for LSTM
energy_data = df['energy_consumption'].values.reshape(-1, 1)
scaler_lstm = MinMaxScaler()
energy_scaled = scaler_lstm.fit_transform(energy_data)

seq_length = 24  # Use 24 hours to predict next hour
X_seq, y_seq = create_sequences(energy_scaled, seq_length)

# Split data
split_idx = int(0.8 * len(X_seq))
X_train_lstm = X_seq[:split_idx]
X_test_lstm = X_seq[split_idx:]
y_train_lstm = y_seq[:split_idx]
y_test_lstm = y_seq[split_idx:]

print(f"\n📊 LSTM Training set size: {len(X_train_lstm)} sequences")
print(f"📊 LSTM Test set size: {len(X_test_lstm)} sequences")

# Build LSTM model
print("\n🤖 Building LSTM model...")
model = keras.Sequential([
    layers.LSTM(64, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\n🔄 Training LSTM model (this may take a minute)...")
history = model.fit(
    X_train_lstm, y_train_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

# Predictions
y_pred_lstm = model.predict(X_test_lstm, verbose=0)

# Inverse transform predictions
y_pred_lstm_original = scaler_lstm.inverse_transform(y_pred_lstm)
y_test_lstm_original = scaler_lstm.inverse_transform(y_test_lstm)

# Evaluate
lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm_original, y_pred_lstm_original))
lstm_r2 = r2_score(y_test_lstm_original, y_pred_lstm_original)
lstm_mae = mean_absolute_error(y_test_lstm_original, y_pred_lstm_original)

print(f"\n✅ LSTM Model Training Complete!")
print(f"\n📈 LSTM Performance:")
print(f"   Test RMSE: {lstm_rmse:.4f} kWh")
print(f"   Test R²: {lstm_r2:.4f}")
print(f"   Test MAE: {lstm_mae:.4f} kWh")

# Save model
model.save('lstm_model.h5')
print("\n✅ Model saved as 'lstm_model.h5'")

# Save predictions
results_lstm = pd.DataFrame({
    'actual': y_test_lstm_original.flatten(),
    'predicted': y_pred_lstm_original.flatten(),
    'error': y_test_lstm_original.flatten() - y_pred_lstm_original.flatten()
})
results_lstm.to_csv('lstm_predictions.csv', index=False)
print("✅ Predictions saved as 'lstm_predictions.csv'")
