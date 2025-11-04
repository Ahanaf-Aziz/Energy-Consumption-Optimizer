import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

print("\n" + "=" * 60)
print("STEP 2: TRAINING RANDOM FOREST MODEL")
print("=" * 60)

# Load data
df = pd.read_csv('energy_data.csv')

# Prepare features and target
X = df[['temperature', 'humidity', 'hour', 'day_of_week', 'month', 'is_weekend']]
y = df['energy_consumption']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Training set size: {len(X_train)} samples")
print(f"📊 Test set size: {len(X_test)} samples")

# Train Random Forest
print("\n🤖 Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluate
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n✅ Model Training Complete!")
print(f"\n📈 Random Forest Performance:")
print(f"   Train RMSE: {train_rmse:.4f} kWh")
print(f"   Test RMSE: {test_rmse:.4f} kWh")
print(f"   Train R²: {train_r2:.4f}")
print(f"   Test R²: {test_r2:.4f}")
print(f"   Test MAE: {test_mae:.4f} kWh")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Top Features by Importance:")
for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Save model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\n✅ Model saved as 'rf_model.pkl'")

# Save predictions
results_rf = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred_test,
    'error': y_test.values - y_pred_test
})
results_rf.to_csv('rf_predictions.csv', index=False)
print("✅ Predictions saved as 'rf_predictions.csv'")
