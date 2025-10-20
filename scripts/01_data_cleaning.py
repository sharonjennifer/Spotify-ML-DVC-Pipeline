import pandas as pd
import numpy as np

# Load raw data
print("Loading raw Spotify data...")
df = pd.read_csv('data/spotify_songs.csv')

print(f"Original dataset shape: {df.shape}")
print(f"\nFirst few columns: {df.columns[:10].tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Data Cleaning
print("\n--- Starting Data Cleaning ---")

# 1. Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} duplicate rows")

# 2. Handle missing values
df = df.dropna()
print(f"Removed rows with missing values. New shape: {df.shape}")

# Save cleaned data
output_path = 'data/spotify_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Cleaned data saved to: {output_path}")
print(f"Final dataset shape: {df.shape}")