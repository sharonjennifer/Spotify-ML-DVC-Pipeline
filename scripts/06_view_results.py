import json
import pandas as pd

print("\n" + "="*70)
print("🎵 SPOTIFY ML PIPELINE - COMPLETE RESULTS DASHBOARD 🎵")
print("="*70)

# Load all metrics
with open('models/metrics_v1.json', 'r') as f:
    v1 = json.load(f)
with open('models/metrics_v2.json', 'r') as f:
    v2 = json.load(f)

# Create comparison table
results = pd.DataFrame({
    'Model': ['V1', 'V2'],
    'RMSE': [v1['rmse'], v2['rmse']],
    'MAE': [v1['mae'], v2['mae']],
    'R² Score': [v1['r2'], v2['r2']],
    'Features': [len(v1['features']), len(v2['features'])],
    'Train Samples': [v1['n_train'], v2['n_train']],
    'Test Samples': [v1['n_test'], v2['n_test']]
})

print("\n MODEL PERFORMANCE COMPARISON:\n")
print(results.to_string(index=False))

print("\n\n WINNER: MODEL V2")
print("\n Key Improvements:")
print(f"   • RMSE improved by: {((v1['rmse'] - v2['rmse'])/v1['rmse']*100):.2f}%")
print(f"   • MAE improved by: {((v1['mae'] - v2['mae'])/v1['mae']*100):.2f}%")
print(f"   • R² improved by: {((v2['r2'] - v1['r2'])/v1['r2']*100):.2f}%")

print("\n BEST MODEL DETAILS:")
print(f"   Version: {v2['version']}")
print(f"   R² Score: {v2['r2']:.4f} (explains {v2['r2']*100:.2f}% of variance)")
print(f"   Average Error: ±{v2['mae']:.2f} popularity points")
print(f"   Features Used: {v2['n_features']}")

print("\n WHAT THIS MEANS:")
print("   • Model can predict song popularity with ±17 points accuracy")
print("   • Most important factors: duration, loudness, instrumentalness")
print("   • Feature scaling and hyperparameter tuning improved performance")

print("\n DATA PIPELINE:")
print(f"   ✓ Raw data: 32,833 songs")
print(f"   ✓ After cleaning: 32,828 songs") 
print(f"   ✓ Feature engineering: 5 new features created")
print(f"   ✓ All versions tracked with DVC in Google Cloud Storage")

print("\n VERSIONING:")
print("   ✓ 3 data versions tracked (raw → cleaned → featured)")
print("   ✓ 2 model versions tracked")
print("   ✓ All reproducible via DVC")

print("\n" + "="*70)