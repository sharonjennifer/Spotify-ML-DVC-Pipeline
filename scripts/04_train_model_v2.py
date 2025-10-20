import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
from datetime import datetime

print("=" * 50)
print("MODEL V2 - WITH FEATURE SCALING")
print("=" * 50)

# Load featured data
print("\nLoading featured data...")
df = pd.read_csv('data/spotify_featured.csv')

# Enhanced feature set (adding more features)
feature_cols = ['danceability', 'energy', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo_normalized',
                'energy_ratio', 'mood_score', 'duration_min',
                'key', 'mode']  # Added 2 more features

target_col = 'track_popularity'

X = df[feature_cols]
y = df[target_col]

print(f"Features shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# NEW: Apply feature scaling
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with different hyperparameters
print("\nTraining Random Forest model (v2)...")
model = RandomForestRegressor(
    n_estimators=150,  # More trees
    max_depth=20,      # Deeper trees
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✓ Model trained!")

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel V2 Performance:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")

# Save model AND scaler
model_path = 'models/spotify_model_v2.pkl'
scaler_path = 'models/scaler_v2.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n✓ Model saved to: {model_path}")
print(f"✓ Scaler saved to: {scaler_path}")

# Save metrics
metrics = {
    'version': 'v2',
    'timestamp': datetime.now().isoformat(),
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2),
    'features': feature_cols,
    'n_features': len(feature_cols),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'improvements': 'Added StandardScaler, 2 more features, tuned hyperparameters'
}

metrics_path = 'models/metrics_v2.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 50)
print("MODEL V2 TRAINING COMPLETE!")
print("=" * 50)