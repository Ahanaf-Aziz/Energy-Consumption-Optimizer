import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 60)
print("STEP 5: MODEL COMPARISON & VISUALIZATION")
print("=" * 60)

# Load results
rf_results = pd.read_csv('rf_predictions.csv')
lstm_results = pd.read_csv('lstm_predictions.csv')

# Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

rf_rmse = np.sqrt(mean_squared_error(rf_results['actual'], rf_results['predicted']))
rf_r2 = r2_score(rf_results['actual'], rf_results['predicted'])
rf_mae = mean_absolute_error(rf_results['actual'], rf_results['predicted'])

lstm_rmse = np.sqrt(mean_squared_error(lstm_results['actual'], lstm_results['predicted']))
lstm_r2 = r2_score(lstm_results['actual'], lstm_results['predicted'])
lstm_mae = mean_absolute_error(lstm_results['actual'], lstm_results['predicted'])

print("\n📊 MODEL COMPARISON:")
print(f"\n{'Metric':<15} {'Random Forest':<20} {'LSTM':<20}")
print("-" * 55)
print(f"{'RMSE':<15} {rf_rmse:<20.4f} {lstm_rmse:<20.4f}")
print(f"{'R² Score':<15} {rf_r2:<20.4f} {lstm_r2:<20.4f}")
print(f"{'MAE':<15} {rf_mae:<20.4f} {lstm_mae:<20.4f}")

# Determine best model
best_model = "Random Forest" if rf_r2 > lstm_r2 else "LSTM"
print(f"\n🏆 Best Model: {best_model}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Energy Consumption Prediction Results', fontsize=16, fontweight='bold')

# Plot 1: RF Predictions vs Actual
axes[0, 0].scatter(rf_results['actual'], rf_results['predicted'], alpha=0.5, s=20)
axes[0, 0].plot([rf_results['actual'].min(), rf_results['actual'].max()], 
                [rf_results['actual'].min(), rf_results['actual'].max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Energy (kWh)')
axes[0, 0].set_ylabel('Predicted Energy (kWh)')
axes[0, 0].set_title(f'Random Forest (R²={rf_r2:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: LSTM Predictions vs Actual
axes[0, 1].scatter(lstm_results['actual'], lstm_results['predicted'], alpha=0.5, s=20, color='orange')
axes[0, 1].plot([lstm_results['actual'].min(), lstm_results['actual'].max()], 
                [lstm_results['actual'].min(), lstm_results['actual'].max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Energy (kWh)')
axes[0, 1].set_ylabel('Predicted Energy (kWh)')
axes[0, 1].set_title(f'LSTM (R²={lstm_r2:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Error Distribution
axes[1, 0].hist(rf_results['error'], bins=30, alpha=0.6, label='Random Forest', color='blue')
axes[1, 0].hist(lstm_results['error'], bins=30, alpha=0.6, label='LSTM', color='orange')
axes[1, 0].set_xlabel('Prediction Error (kWh)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Error Distribution Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Model Metrics Comparison
metrics_data = {
    'RMSE': [rf_rmse, lstm_rmse],
    'R² Score': [rf_r2, lstm_r2],
    'MAE': [rf_mae, lstm_mae]
}
x = np.arange(len(metrics_data))
width = 0.35

for i, (metric, values) in enumerate(metrics_data.items()):
    axes[1, 1].bar(i - width/2, values[0], width, label='RF' if i == 0 else '')
    axes[1, 1].bar(i + width/2, values[1], width, label='LSTM' if i == 0 else '')

axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Performance Metrics')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics_data.keys())
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Comparison chart saved as 'model_comparison.png'")
plt.show()

print("\n" + "=" * 60)
print("✅ COMPLETE PIPELINE EXECUTION FINISHED!")
print("=" * 60)
print("\nGenerated files:")
print("  • energy_data.csv - Preprocessed dataset")
print("  • rf_model.pkl - Trained Random Forest model")
print("  • lstm_model.h5 - Trained LSTM model")
print("  • rf_predictions.csv - RF predictions")
print("  • lstm_predictions.csv - LSTM predictions")
print("  • optimization_strategies.csv - Energy saving strategies")
print("  • model_comparison.png - Visualization chart")
print("\n🎉 Project Complete!")
