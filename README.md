# Spotify Song Popularity Prediction using DVC

## Project Overview

This project implements a machine learning pipeline with Data Version Control (DVC) to predict Spotify song popularity. The project demonstrates data versioning, model tracking, and experiment management using DVC integrated with Google Cloud Storage.

## Dataset

- **Source:** Kaggle - Spotify Songs Dataset
- **Size:** 32,828 songs after cleaning
- **Features:** 23 audio characteristics including danceability, energy, tempo, acousticness, loudness, valence, etc.
- **Target:** track_popularity (0-100 scale)

## Pipeline Architecture

The project consists of 6 stages, with data and models tracked using DVC:

### Stage 1: Data Cleaning
- **Script:** `01_data_cleaning.py`
- **Input:** Raw spotify_songs.csv (32,833 songs)
- **Process:** Remove duplicates and handle missing values
- **Output:** spotify_cleaned.csv (32,828 songs)
- **DVC Tracked:** Yes

### Stage 2: Feature Engineering
- **Script:** `02_feature_engineering.py`
- **Input:** spotify_cleaned.csv
- **Process:** Create 5 new features
  - energy_ratio (energy/acousticness ratio)
  - mood_score (valence × danceability)
  - tempo_normalized (scaled 0-1)
  - duration_min (duration in minutes)
  - popularity_category (Low/Medium/High)
- **Output:** spotify_featured.csv (28 columns)
- **DVC Tracked:** Yes

### Stage 3: Model Training V1
- **Script:** `03_train_model.py`
- **Algorithm:** Random Forest Regressor
- **Parameters:** 100 trees, max_depth=15
- **Features:** 12 audio features
- **Output:** spotify_model_v1.pkl, metrics_v1.json
- **DVC Tracked:** Yes

### Stage 4: Model Training V2
- **Script:** `04_train_model_v2.py`
- **Algorithm:** Random Forest Regressor (improved)
- **Improvements:**
  - StandardScaler for feature normalization
  - 14 features (added key and mode)
  - Tuned hyperparameters (150 trees, max_depth=20)
- **Output:** spotify_model_v2.pkl, scaler_v2.pkl, metrics_v2.json
- **DVC Tracked:** Yes

### Stage 5: Model Comparison
- **Script:** `05_compare_models.py`
- **Purpose:** Side-by-side comparison of model versions

### Stage 6: Results Dashboard
- **Script:** `06_view_results.py`
- **Purpose:** Comprehensive results summary

## Results

### Model Performance

| Metric | Model V1 | Model V2 | Change |
|--------|----------|----------|--------|
| RMSE | 21.69 | 21.27 | -1.92% |
| MAE | 17.73 | 16.99 | -4.21% |
| R² Score | 0.2335 | 0.2627 | +12.50% |
| Features | 12 | 14 | +2 |

Model V2 achieves better performance with an R² score of 0.2627, explaining 26.27% of variance in song popularity with an average error of ±17 popularity points.

### Feature Importance (Model V1)
1. duration_min
2. loudness
3. instrumentalness
4. tempo_normalized
5. energy

## Project Structure
```
Spotify_ML_Pipeline/
├── data/
│   ├── spotify_songs.csv          # Raw data [DVC tracked]
│   ├── spotify_cleaned.csv        # Cleaned data [DVC tracked]
│   └── spotify_featured.csv       # Featured data [DVC tracked]
├── models/
│   ├── spotify_model_v1.pkl       # Model V1 [DVC tracked]
│   ├── spotify_model_v2.pkl       # Model V2 [DVC tracked]
│   ├── scaler_v2.pkl              # Feature scaler [DVC tracked]
│   ├── metrics_v1.json            # Model V1 metrics
│   └── metrics_v2.json            # Model V2 metrics
├── scripts/
│   ├── 01_data_cleaning.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_model.py
│   ├── 04_train_model_v2.py
│   ├── 05_compare_models.py
│   └── 06_view_results.py
├── .dvc/                          # DVC configuration
├── .git/                          # Git repository
└── README.md
```

## Technologies Used

- **Python 3.x**
- **DVC** - Data Version Control
- **Google Cloud Storage** - Remote storage for data and models
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **numpy** - Numerical computing

## Setup and Installation

### Prerequisites
```bash
pip install dvc[gs] pandas numpy scikit-learn
```

### DVC Configuration
```bash
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d myremote gs://dvc-mlops-lab-jennifer/spotify-pipeline

# Configure credentials
dvc remote modify myremote credentialpath path/to/credentials.json
```

## Running the Pipeline

### Execute All Stages
```bash
# Data processing
python scripts/01_data_cleaning.py
python scripts/02_feature_engineering.py

# Model training
python scripts/03_train_model.py
python scripts/04_train_model_v2.py

# View results
python scripts/05_compare_models.py
python scripts/06_view_results.py
```

### Track Changes with DVC
```bash
# Add data/model files to DVC
dvc add data/spotify_songs.csv
dvc add models/spotify_model_v1.pkl

# Commit metadata to Git
git add data/spotify_songs.csv.dvc
git commit -m "Track dataset with DVC"

# Push to remote storage
dvc push
```

### Pull Data and Models
```bash
# Download all tracked files from cloud
dvc pull
```

## DVC Workflow

This project demonstrates key DVC operations:

1. **Version Control:** Track multiple versions of datasets and models
2. **Remote Storage:** Store large files in Google Cloud Storage
3. **Reproducibility:** Anyone can pull exact versions used in experiments
4. **Collaboration:** Share data and models across team members
5. **Experiment Tracking:** Compare different model versions with metrics

## Key DVC Commands Used

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in the project |
| `dvc add <file>` | Start tracking a file with DVC |
| `dvc push` | Upload tracked files to remote storage |
| `dvc pull` | Download tracked files from remote storage |
| `dvc status -c` | Check sync status with remote |
| `dvc checkout` | Switch to a specific file version |

## Reproducibility

To reproduce results:

1. Clone the repository
2. Configure DVC remote with credentials
3. Run `dvc pull` to download data and models
4. Execute pipeline scripts in order
5. Results will match the metrics in `models/metrics_*.json`

## Author

Sharon Jennifer Justin Devaraj  
MLOps Lab - DVC Implementation  


