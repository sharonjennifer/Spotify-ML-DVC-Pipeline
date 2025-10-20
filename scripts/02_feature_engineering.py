import pandas as pd
import numpy as np

# Load cleaned data
print("Loading cleaned Spotify data...")
df = pd.read_csv('data/spotify_cleaned.csv')

print(f"Dataset shape: {df.shape}")

# Feature Engineering
print("\n--- Feature Engineering ---")

# 1. Create energy ratio (high energy + low acousticness = energetic songs)
df['energy_ratio'] = df['energy'] / (df['acousticness'] + 0.01)

# 2. Create mood score (valence * danceability)
df['mood_score'] = df['valence'] * df['danceability']

# 3. Normalize tempo to 0-1 scale
df['tempo_normalized'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())

# 4. Convert duration from ms to minutes
df['duration_min'] = df['duration_ms'] / 60000

# 5. Create popularity categories for easier modeling
df['popularity_category'] = pd.cut(df['track_popularity'], 
                                     bins=[0, 30, 60, 100], 
                                     labels=['Low', 'Medium', 'High'])

print(f"\nNew features created:")
print(f"- energy_ratio")
print(f"- mood_score")
print(f"- tempo_normalized")
print(f"- duration_min")
print(f"- popularity_category")

# Select features for modeling
feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo', 'duration_min',
                'energy_ratio', 'mood_score', 'tempo_normalized']

target_col = 'track_popularity'

# Save featured data
output_path = 'data/spotify_featured.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Featured data saved to: {output_path}")
print(f"Final shape: {df.shape}")