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
    model_path = '/app/model_lay_aoav_backtest.joblib' # Updated model path
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
    # Expand cols_to_keep to include B365 odds for feature engineering
    cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                    'B365H', 'B365D', 'B365A', 'LayAOAV']
    cols_present = [col for col in cols_to_keep if col in historical_df.columns] # Keep only available essential cols
    historical_df = historical_df[cols_present]

    # Ensure B365 odds are numeric before they are used for new_matches_sample_df
    odds_cols_to_convert = ['B365H', 'B365D', 'B365A']
    for col in odds_cols_to_convert:
        if col in historical_df.columns:
            historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')
        else:
            print(f"Warning: Odds column {col} not found in historical_df. Predictions may fail or be inaccurate.")
            historical_df[col] = np.nan # Add missing column as NaN to prevent KeyErrors later

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
    new_matches_sample_df = historical_df.iloc[-10:].copy()
    print(f"Simulating prediction for the last {len(new_matches_sample_df)} matches from the dataset.")

    # 4. Feature Engineering for New Matches (Odds-based features)
    # Ensure essential odds columns are present and numeric before calculations
    odds_cols = ['B365H', 'B365D', 'B365A']
    for col in odds_cols:
        if col not in new_matches_sample_df.columns:
            print(f"Error: Critical odds column {col} is missing in new_matches_sample_df. Cannot engineer features.")
            exit() # Exit if essential odds columns are missing
        new_matches_sample_df[col] = pd.to_numeric(new_matches_sample_df[col], errors='coerce')

    # Drop rows if B365H, B365D, or B365A are NaN after conversion, as they are essential for feature engineering
    new_matches_sample_df.dropna(subset=odds_cols, inplace=True)
    if new_matches_sample_df.empty:
        print("No matches left after dropping NaNs from essential odds columns. Cannot make predictions.")
        exit()

    print("Calculating odds-based features for new matches...")
    # 1. Implied Probabilities
    new_matches_sample_df['ProbH'] = 1 / new_matches_sample_df['B365H']
    new_matches_sample_df['ProbD'] = 1 / new_matches_sample_df['B365D']
    new_matches_sample_df['ProbA'] = 1 / new_matches_sample_df['B365A']

    # 2. Normalized Probabilities
    new_matches_sample_df['TotalProb'] = new_matches_sample_df['ProbH'] + new_matches_sample_df['ProbD'] + new_matches_sample_df['ProbA']
    new_matches_sample_df['NormProbH'] = new_matches_sample_df['ProbH'] / new_matches_sample_df['TotalProb']
    new_matches_sample_df['NormProbD'] = new_matches_sample_df['ProbD'] / new_matches_sample_df['TotalProb']
    new_matches_sample_df['NormProbA'] = new_matches_sample_df['ProbA'] / new_matches_sample_df['TotalProb']

    # 3. Bookmaker's Margin
    new_matches_sample_df['Margin'] = (new_matches_sample_df['TotalProb'] - 1) * 100

    # 4. Odds Spreads/Ratios
    new_matches_sample_df['SpreadHA'] = new_matches_sample_df['B365H'] - new_matches_sample_df['B365A']
    new_matches_sample_df['RatioHA'] = new_matches_sample_df['B365H'] / (new_matches_sample_df['B365A'] + 1e-6) # Epsilon for safety

    # 5. Log Odds (ensure positivity)
    new_matches_sample_df['LogOddsH'] = np.log(new_matches_sample_df['B365H'].apply(lambda x: x if x > 0 else 1e-6))
    new_matches_sample_df['LogOddsD'] = np.log(new_matches_sample_df['B365D'].apply(lambda x: x if x > 0 else 1e-6))
    new_matches_sample_df['LogOddsA'] = np.log(new_matches_sample_df['B365A'].apply(lambda x: x if x > 0 else 1e-6))
    print(f"Odds-based features calculated. Shape of new_matches_sample_df: {new_matches_sample_df.shape}")

    # Remove or comment out get_rolling_features and its usage
    # print("Calculating features for the full historical dataset...")
    # all_data_featured = get_rolling_features(historical_df.copy(), N=5) # Use .copy()
    # new_matches_featured = all_data_featured.iloc[-len(new_matches_sample_df):].copy()
    # key_feature_cols_for_nan_check = [
    #     'HomeTeam_RecentPoints', 'AwayTeam_RecentPoints'
    # ]
    # new_matches_featured.dropna(subset=key_feature_cols_for_nan_check, inplace=True)
    # if new_matches_featured.empty:
    #     print("No new matches left to predict after feature engineering and NaN drop...")
    #     exit()
    # print(f"Features calculated for new matches. Shape of new_matches_featured: {new_matches_featured.shape}")
    new_matches_featured = new_matches_sample_df # Use the dataframe with new odds features

    # 5. Make Predictions
    feature_cols = [
        'B365H', 'B365D', 'B365A',
        'ProbH', 'ProbD', 'ProbA', 'TotalProb',
        'NormProbH', 'NormProbD', 'NormProbA', 'Margin',
        'SpreadHA', 'RatioHA',
        'LogOddsH', 'LogOddsD', 'LogOddsA'
    ]

    # Ensure all feature columns are present in the new_matches_featured dataframe
    missing_cols = [col for col in feature_cols if col not in new_matches_featured.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from new_matches_featured: {missing_cols}. These should have been engineered. Cannot make predictions.")
        exit()

    X_new = new_matches_featured[feature_cols]

    # Final check for NaNs in X_new. Odds features should be robust, but this is a safeguard.
    if X_new.isnull().sum().any().any():
        print("Warning: NaNs detected in X_new just before prediction. This indicates an issue in feature engineering or data.")
        # Option: fill with a placeholder if appropriate, or exit
        # X_new.fillna(X_new.mean(), inplace=True) # Example: fill with mean, or 0
        print("Dropping rows with NaNs in features before prediction.")
        nan_rows_before = X_new.isnull().any(axis=1)
        X_new = X_new.dropna()
        new_matches_featured = new_matches_featured[~nan_rows_before] # Keep corresponding rows in new_matches_featured
        if X_new.empty:
            print("No data left after dropping NaNs from X_new. Cannot make predictions.")
            exit()

    if X_new.empty:
        print("X_new is empty before predictions. Cannot proceed.")
        exit()

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    # 6. Display Predictions
    print("\n--- Predictions for Simulated New Matches (Odds-based Model) ---")
    # Iterate using index from new_matches_featured to ensure correct row alignment
    # This is critical if X_new had rows dropped due to NaNs
    for i, index_val in enumerate(new_matches_featured.index):
        # Check if index_val is still in X_new (if rows were dropped from X_new but not new_matches_featured initially)
        # This check is somewhat redundant now as new_matches_featured is filtered alongside X_new if NaNs are dropped.
        if index_val not in X_new.index: # Should not happen if new_matches_featured is filtered with X_new
            continue

        match_details = new_matches_featured.loc[index_val]

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
