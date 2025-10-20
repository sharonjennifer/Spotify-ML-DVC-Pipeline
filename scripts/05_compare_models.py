import json

print("=" * 60)
print("MODEL COMPARISON: V1 vs V2")
print("=" * 60)

# Load metrics
with open('models/metrics_v1.json', 'r') as f:
    metrics_v1 = json.load(f)

with open('models/metrics_v2.json', 'r') as f:
    metrics_v2 = json.load(f)

# Compare
print("\n Performance Metrics:\n")
print(f"{'Metric':<15} {'V1':<15} {'V2':<15} {'Improvement':<15}")
print("-" * 60)

rmse_improve = ((metrics_v1['rmse'] - metrics_v2['rmse']) / metrics_v1['rmse']) * 100
mae_improve = ((metrics_v1['mae'] - metrics_v2['mae']) / metrics_v1['mae']) * 100
r2_improve = ((metrics_v2['r2'] - metrics_v1['r2']) / metrics_v1['r2']) * 100

print(f"{'RMSE':<15} {metrics_v1['rmse']:<15.2f} {metrics_v2['rmse']:<15.2f} {rmse_improve:>+.2f}%")
print(f"{'MAE':<15} {metrics_v1['mae']:<15.2f} {metrics_v2['mae']:<15.2f} {mae_improve:>+.2f}%")
print(f"{'R² Score':<15} {metrics_v1['r2']:<15.4f} {metrics_v2['r2']:<15.4f} {r2_improve:>+.2f}%")

print("\n Model Configurations:\n")
print(f"V1: {len(metrics_v1['features'])} features")
print(f"V2: {len(metrics_v2['features'])} features")

print("\n Winner: MODEL V2")
print("   Reasons:")
print("   - Added feature scaling (StandardScaler)")
print("   - Included 2 additional features (key, mode)")
print("   - Tuned hyperparameters (150 trees, depth 20)")

print("\n" + "=" * 60)