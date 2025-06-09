import os
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from collections import deque # Needed for get_rolling_features if not using lists

# --- Helper Functions (from feature_utils.py) ---
def get_match_points(result_indicator, perspective):
    if perspective == 'home':
        if result_indicator == 'H': return 3
        if result_indicator == 'D': return 1
        return 0
    elif perspective == 'away':
        if result_indicator == 'A': return 3
        if result_indicator == 'D': return 1
        return 0
    return 0

def get_rolling_features(df, N=5): # Removed existing_home_stats, existing_away_stats for simplicity in consolidation
    df_copy = df.sort_values(by='Date').copy()

    home_team_stats = {}
    away_team_stats = {}

    feature_names = [
        'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
        'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded'
    ]
    for col in feature_names:
        df_copy[col] = np.nan

    all_teams = pd.concat([df_copy['HomeTeam'], df_copy['AwayTeam']]).unique()
    for team in all_teams:
        # Use deque for efficient append and pop from left
        home_team_stats[team] = deque(maxlen=N)
        away_team_stats[team] = deque(maxlen=N)

    for index, row in df_copy.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Calculate for Home Team (must have N previous games in its deque)
        if len(home_team_stats[home_team]) == N:
            past_N_games = list(home_team_stats[home_team]) # Convert deque to list for sum
            df_copy.loc[index, 'HomeTeam_RecentPoints'] = sum(g['points'] for g in past_N_games)
            df_copy.loc[index, 'HomeTeam_RecentGoalsScored'] = sum(g['goals_scored'] for g in past_N_games)
            df_copy.loc[index, 'HomeTeam_RecentGoalsConceded'] = sum(g['goals_conceded'] for g in past_N_games)

        # Calculate for Away Team (must have N previous games in its deque)
        if len(away_team_stats[away_team]) == N:
            past_N_games = list(away_team_stats[away_team]) # Convert deque to list for sum
            df_copy.loc[index, 'AwayTeam_RecentPoints'] = sum(g['points'] for g in past_N_games)
            df_copy.loc[index, 'AwayTeam_RecentGoalsScored'] = sum(g['goals_scored'] for g in past_N_games)
            df_copy.loc[index, 'AwayTeam_RecentGoalsConceded'] = sum(g['goals_conceded'] for g in past_N_games)

        home_points = get_match_points(row['FTR'], 'home')
        home_team_stats[home_team].append({'points': home_points, 'goals_scored': row['FTHG'], 'goals_conceded': row['FTAG']})

        away_points = get_match_points(row['FTR'], 'away')
        away_team_stats[away_team].append({'points': away_points, 'goals_scored': row['FTAG'], 'goals_conceded': row['FTHG']})

    return df_copy

# --- Main Script Logic ---
print("Starting consolidated model trainer script...")

# 1. Data Downloading
print("\n--- Section 1: Downloading Data ---")
BASE_URL = "https://www.football-data.co.uk/mmz4281/"
SEASONS = ['2324', '2223', '2122']
LEAGUE_CODES = ['E0']
downloaded_csv_paths = []
temp_dir = "/tmp" # Save downloads to /tmp

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir) # Should exist in this environment

for league_code in LEAGUE_CODES:
    for season in SEASONS:
        url = f"{BASE_URL}{season}/{league_code}.csv"
        filepath = os.path.join(temp_dir, f"{league_code}_{season}.csv")
        print(f"Attempting to download data from {url} to {filepath}")
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            downloaded_csv_paths.append(filepath)
            print(f"Successfully downloaded and saved {filepath}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")

if not downloaded_csv_paths:
    print("No CSV files were downloaded. Exiting.")
    exit()

# 2. Data Preprocessing
print("\n--- Section 2: Preprocessing Data ---")
dataframes = []
for path in downloaded_csv_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
            print(f"Successfully read {path}")
        except Exception as e:
            print(f"Could not read {path}: {e}")
    else:
        print(f"File {path} not found after download (unexpected).")


if not dataframes:
    print("No dataframes loaded from downloaded files. Exiting.")
    exit()

combined_df = pd.concat(dataframes, ignore_index=True)
print(f"Combined dataframes. Total rows: {len(combined_df)}")

combined_df['FTAG'] = pd.to_numeric(combined_df['FTAG'], errors='coerce')
combined_df.dropna(subset=['FTAG', 'FTR'], inplace=True) # Critical for LayAOAV

condition_lose_lay = (combined_df['FTR'].astype(str) == 'A') & (combined_df['FTAG'] >= 4)
combined_df['LayAOAV'] = np.where(condition_lose_lay, 0, 1)
print("Created 'LayAOAV' target variable.")

cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'LayAOAV']
# Filter to only existing columns to avoid errors if a source CSV is unusual
cols_present = [col for col in cols_to_keep if col in combined_df.columns]
combined_df = combined_df[cols_present]

# Ensure FTHG is numeric before potential int conversion
combined_df['FTHG'] = pd.to_numeric(combined_df['FTHG'], errors='coerce')
combined_df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG'], inplace=True) # FTAG, FTR already handled

combined_df['FTHG'] = combined_df['FTHG'].astype(int)
combined_df['FTAG'] = combined_df['FTAG'].astype(int)

combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce', dayfirst=True)
combined_df.dropna(subset=['Date'], inplace=True)
processed_df = combined_df.sort_values(by='Date').copy()
print("Data preprocessing complete.")
print(f"Processed data shape: {processed_df.shape}")
if 'LayAOAV' in processed_df.columns:
    print("Value counts for 'LayAOAV':")
    print(processed_df['LayAOAV'].value_counts(normalize=True))
else:
    print("'LayAOAV' column not found after processing.")


# 3. Feature Engineering
print("\n--- Section 3: Feature Engineering ---")
if processed_df.empty:
    print("Processed dataframe is empty. Cannot proceed to feature engineering.")
    exit()

df_featured = get_rolling_features(processed_df, N=5) # Use .copy() if processed_df is used later
print(f"Applied rolling features. Shape: {df_featured.shape}")
df_featured.dropna(inplace=True)
print(f"Dropped NaNs after rolling features. Shape: {df_featured.shape}")

if df_featured.empty:
    print("DataFrame is empty after feature engineering and NaN drop. Cannot train model.")
    exit()

# 4. Model Training
print("\n--- Section 4: Model Training ---")
feature_cols = ['HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
                'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded']
target_col = 'LayAOAV'

# Check if all expected columns are present
if not all(col in df_featured.columns for col in feature_cols):
    print(f"Error: Not all feature columns are present. Missing: {[col for col in feature_cols if col not in df_featured.columns]}")
    exit()
if target_col not in df_featured.columns:
    print(f"Error: Target column '{target_col}' not found.")
    exit()

X = df_featured[feature_cols]
y = df_featured[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# MODIFIED LINE: Added class_weight='balanced'
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
print("Model trained successfully (with class_weight='balanced').")

# 5. Model Evaluation
# MODIFIED LINE: Updated evaluation header
print("\n--- Section 5: Model Evaluation (with class_weight='balanced') ---")
if X_test.empty:
    print("X_test is empty, skipping evaluation.")
else:
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix:\n", cm)
    report = classification_report(y_test, y_pred_test)
    print("Classification Report:\n", report)
    roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    print(f"ROC AUC Score: {roc_auc:.4f}")

# 6. Save Model
print("\n--- Section 6: Saving Model ---")
model_filename = './app/model_lay_aoav.joblib' # Saving in /app for persistence if environment allows
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

print("\nConsolidated model trainer script finished.")
