import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
from datetime import datetime

print("=" * 50)
print("SPOTIFY POPULARITY PREDICTION - MODEL TRAINING")
print("=" * 50)

# Load featured data
print("\nLoading featured data...")
df = pd.read_csv('data/spotify_featured.csv')

# Select features for modeling
feature_cols = ['danceability', 'energy', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo_normalized',
                'energy_ratio', 'mood_score', 'duration_min']

target_col = 'track_popularity'

X = df[feature_cols]
y = df[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✓ Model trained!")

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Most Important Features:")
print(feature_importance.head())

# Save model
model_path = 'models/spotify_model_v1.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\n✓ Model saved to: {model_path}")

# Save metrics
metrics = {
    'version': 'v1',
    'timestamp': datetime.now().isoformat(),
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2),
    'features': feature_cols,
    'n_train': len(X_train),
    'n_test': len(X_test)
}

metrics_path = 'models/metrics_v1.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)