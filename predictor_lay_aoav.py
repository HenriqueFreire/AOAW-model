import os
import requests
import pandas as pd
import numpy as np
import joblib
from collections import deque # For get_rolling_features

# --- Helper Functions (copied from model_trainer.py for self-containment) ---
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

def get_rolling_features(df, N=5, existing_home_stats=None, existing_away_stats=None):
    # existing_home_stats and existing_away_stats are not used in this simplified version
    # but kept in signature for potential future compatibility if needed.
    # This version recalculates stats from the provided df.
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
        home_team_stats[team] = deque(maxlen=N)
        away_team_stats[team] = deque(maxlen=N)

    for index, row in df_copy.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        if len(home_team_stats[home_team]) == N:
            past_N_games = list(home_team_stats[home_team])
            df_copy.loc[index, 'HomeTeam_RecentPoints'] = sum(g['points'] for g in past_N_games)
            df_copy.loc[index, 'HomeTeam_RecentGoalsScored'] = sum(g['goals_scored'] for g in past_N_games)
            df_copy.loc[index, 'HomeTeam_RecentGoalsConceded'] = sum(g['goals_conceded'] for g in past_N_games)

        if len(away_team_stats[away_team]) == N:
            past_N_games = list(away_team_stats[away_team])
            df_copy.loc[index, 'AwayTeam_RecentPoints'] = sum(g['points'] for g in past_N_games)
            df_copy.loc[index, 'AwayTeam_RecentGoalsScored'] = sum(g['goals_scored'] for g in past_N_games)
            df_copy.loc[index, 'AwayTeam_RecentGoalsConceded'] = sum(g['goals_conceded'] for g in past_N_games)

        home_points = get_match_points(row['FTR'], 'home')
        home_team_stats[home_team].append({'points': home_points, 'goals_scored': row['FTHG'], 'goals_conceded': row['FTAG']})

        away_points = get_match_points(row['FTR'], 'away')
        away_team_stats[away_team].append({'points': away_points, 'goals_scored': row['FTAG'], 'goals_conceded': row['FTHG']})

    return df_copy

# --- Main Prediction Logic ---
if __name__ == '__main__':
    print("Starting predictor_lay_aoav.py script...")

    # 1. Load Model
    model_path = '/app/model_lay_aoav.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first using model_trainer.py.")
        exit()
    try:
        model = joblib.load(model_path)
        print(f"Model {model_path} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        exit()

    # 2. Data Acquisition and Initial Processing (mimicking model_trainer.py)
    print("\n--- Acquiring and Preprocessing Data for Prediction Context ---")
    BASE_URL = "https://www.football-data.co.uk/mmz4281/"
    SEASONS = ['2324', '2223', '2122'] # Use same seasons as training for consistency
    LEAGUE_CODES = ['E0']
    downloaded_csv_paths = []
    temp_dir = "/tmp"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for league_code in LEAGUE_CODES:
        for season in SEASONS:
            url = f"{BASE_URL}{season}/{league_code}.csv"
            filepath = os.path.join(temp_dir, f"pred_dl_{league_code}_{season}.csv") # Prefix to avoid name clash
            print(f"Attempting to download {url} to {filepath}")
            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                downloaded_csv_paths.append(filepath)
                print(f"Successfully downloaded {filepath}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url}: {e}")

    if not downloaded_csv_paths:
        print("No CSV files downloaded for historical context. Exiting.")
        exit()

    dataframes = []
    for path in downloaded_csv_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dataframes.append(df)
            except Exception as e:
                print(f"Could not read {path}: {e}")

    if not dataframes:
        print("No dataframes loaded from downloaded files. Exiting.")
        exit()

    historical_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined historical data. Total rows: {len(historical_df)}")

    # Preprocess historical_df (same as in model_trainer.py)
    historical_df['FTAG'] = pd.to_numeric(historical_df['FTAG'], errors='coerce')
    historical_df.dropna(subset=['FTAG', 'FTR'], inplace=True)
    condition_lose_lay = (historical_df['FTR'].astype(str) == 'A') & (historical_df['FTAG'] >= 4)
    historical_df['LayAOAV'] = np.where(condition_lose_lay, 0, 1)
    cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'LayAOAV']
    cols_present = [col for col in cols_to_keep if col in historical_df.columns] # Keep only available essential cols
    historical_df = historical_df[cols_present]
    historical_df['FTHG'] = pd.to_numeric(historical_df['FTHG'], errors='coerce')
    historical_df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG'], inplace=True)
    historical_df['FTHG'] = historical_df['FTHG'].astype(int)
    historical_df['FTAG'] = historical_df['FTAG'].astype(int) # FTAG already made numeric and NaN-dropped
    historical_df['Date'] = pd.to_datetime(historical_df['Date'], errors='coerce', dayfirst=True)
    historical_df.dropna(subset=['Date'], inplace=True)
    historical_df.sort_values(by='Date', inplace=True)
    print(f"Historical data preprocessed. Shape: {historical_df.shape}")


    # 3. Simulate "New" Matches for Prediction
    if len(historical_df) < 20: # Need enough data for meaningful context + sample
        print("Not enough historical data to create a meaningful sample for prediction after processing. Exiting.")
        exit()

    # Take the last 10 matches as "new" data to predict
    # The features for these will be calculated based on the games before them.
    new_matches_sample_df = historical_df.iloc[-10:].copy()
    print(f"Simulating prediction for the last {len(new_matches_sample_df)} matches from the dataset.")

    # 4. Feature Engineering for New Matches
    # The get_rolling_features function will calculate features for all rows.
    # We pass the full historical_df because the function builds context internally.
    # Then we select the part corresponding to new_matches_sample_df.
    print("Calculating features for the full historical dataset...")
    all_data_featured = get_rolling_features(historical_df.copy(), N=5) # Use .copy()

    # Select the last N rows that correspond to new_matches_sample_df
    # Ensure indices align if historical_df was re-indexed by get_rolling_features (it shouldn't be, but good practice)
    new_matches_featured = all_data_featured.iloc[-len(new_matches_sample_df):].copy()

    # Drop rows if key features (e.g., HomeTeam_RecentPoints) are NaN.
    # This could happen for the very first few games in the overall dataset.
    # For the *last* 10 games, this is less likely if N=5 and dataset is large enough.
    key_feature_cols_for_nan_check = [
        'HomeTeam_RecentPoints', 'AwayTeam_RecentPoints' # Check a key feature for both home and away
    ]
    new_matches_featured.dropna(subset=key_feature_cols_for_nan_check, inplace=True)

    if new_matches_featured.empty:
        print("No new matches left to predict after feature engineering and NaN drop. This might happen if the sample was too small or from the very start of the dataset.")
        exit()
    print(f"Features calculated for new matches. Shape of new_matches_featured: {new_matches_featured.shape}")

    # 5. Make Predictions
    feature_cols = ['HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
                    'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded']

    # Ensure all feature columns are present in the new_matches_featured dataframe
    missing_cols = [col for col in feature_cols if col not in new_matches_featured.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from new_matches_featured: {missing_cols}. Cannot make predictions.")
        exit()

    X_new = new_matches_featured[feature_cols]

    # One final check for NaNs in X_new that might have slipped through
    if X_new.isnull().sum().any().any(): # .any().any() checks if any NaN exists in the entire DataFrame
        print("Warning: NaNs detected in X_new just before prediction. Filling with 0. This should ideally be handled by earlier NaN checks.")
        X_new.fillna(0, inplace=True)


    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    # 6. Display Predictions
    print("\n--- Predictions for Simulated New Matches ---")
    # Iterate using index from new_matches_featured to ensure correct row alignment
    for i, index_val in enumerate(new_matches_featured.index):
        match_details = new_matches_featured.loc[index_val] # Use .loc with the actual index

        home_team = match_details['HomeTeam']
        away_team = match_details['AwayTeam']
        match_date_obj = match_details['Date'] # This should be a datetime object
        match_date_str = match_date_obj.strftime('%Y-%m-%d') if pd.notna(match_date_obj) else "N/A"

        predicted_class = predictions[i]
        # Prob of LayAOAV=1 (class 1)
        proba_lay_aoav = probabilities[i][model.classes_.tolist().index(1)] if 1 in model.classes_ else probabilities[i][0]


        actual_outcome = match_details['LayAOAV']
        actual_ftr = match_details['FTR']
        actual_fthg = match_details['FTHG']
        actual_ftag = match_details['FTAG']

        print(f"Match: {home_team} vs {away_team} on {match_date_str}")
        print(f"  Predicted LayAOAV: {predicted_class} (1=Lay, 0=Don't Lay/Risky)")
        print(f"  Probability of LayAOAV (Visitor NOT Win by 4+ goals): {proba_lay_aoav:.4f}")
        print(f"  Actual LayAOAV: {actual_outcome} (FTR: {actual_ftr}, Score: {actual_fthg}-{actual_ftag})")
        print("  ---")

    print("\npredictor_lay_aoav.py script finished.")
