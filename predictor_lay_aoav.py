import os
import requests
import pandas as pd
import numpy as np
import joblib
# from collections import deque # No longer needed

# Helper functions get_match_points and get_rolling_features are removed as features are pre-calculated.

# --- Main Prediction Logic ---
if __name__ == '__main__':
    print("Starting predictor_lay_aoav.py script with clustered data...")

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

    # 2. Load Historical Clustered Data for Prediction Context
    print("\n--- Loading Historical Clustered Data for Prediction Context ---")
    # This file is expected to be created by cluster_features.py
    # It should contain rolling features and 'match_cluster_id'.
    try:
        historical_df = pd.read_csv('/app/processed_data_clustered.csv', parse_dates=['Date'])
        print(f"Successfully loaded /app/processed_data_clustered.csv. Shape: {historical_df.shape}")
        historical_df.sort_values(by='Date', inplace=True)
    except FileNotFoundError:
        print("Error: /app/processed_data_clustered.csv not found. Please ensure cluster_features.py has run successfully.")
        exit()
    except Exception as e:
        print(f"Error loading /app/processed_data_clustered.csv: {e}")
        exit()

    if historical_df.empty:
        print("Loaded historical_df from clustered data is empty. Exiting.")
        exit()

    # Safety NaN drop (cluster_features.py should have handled NaNs for rolling features,
    # but this is a safeguard for other potential NaNs or if the file is somehow corrupted)
    historical_df.dropna(inplace=True)
    print(f"Shape of historical_df after NaN drop: {historical_df.shape}")

    if historical_df.empty:
        print("Historical_df is empty after NaN drop. Exiting.")
        exit()

    # 3. Simulate "New" Matches for Prediction
    # The historical_df now already contains pre-calculated rolling features and cluster_id
    # We'll call it `all_data_featured` to align with previous naming, though it's just historical_df.
    all_data_featured = historical_df

    if len(all_data_featured) < 10: # Arbitrary small number, ensure enough data for a sample
        print("Not enough historical data to create a meaningful sample for prediction. Exiting.")
        exit()

    # Take the last 10 matches as "new" data to predict
    new_matches_sample_df = all_data_featured.iloc[-10:].copy()
    print(f"Simulating prediction for the last {len(new_matches_sample_df)} matches.")

    if new_matches_sample_df.empty:
        print("The sample of new matches is empty. This should not happen if historical_df has enough data. Exiting.")
        exit()

    # Check if 'match_cluster_id' is present
    if 'match_cluster_id' not in new_matches_sample_df.columns:
        print("Error: 'match_cluster_id' column not found in new_matches_sample_df. Ensure cluster_features.py ran correctly.")
        exit()

    # 4. Prepare Features for "New" Matches (X_new)
    print("Preparing features for new matches (X_new) including cluster dummies...")
    rolling_feature_cols = [
        'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
        'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded'
    ]

    # Verify all rolling feature columns are present
    missing_rolling_features = [col for col in rolling_feature_cols if col not in new_matches_sample_df.columns]
    if missing_rolling_features:
        print(f"Error: Missing rolling feature columns in new_matches_sample_df: {missing_rolling_features}")
        exit()

    X_new = new_matches_sample_df[rolling_feature_cols].copy()

    # One-Hot Encode 'match_cluster_id' for the sample
    # To ensure consistency with training, convert 'match_cluster_id' to categorical
    # using all possible categories from the entire historical_df (which should match training data context).
    all_possible_cluster_ids = sorted(historical_df['match_cluster_id'].unique())
    new_matches_sample_df['match_cluster_id'] = pd.Categorical(
        new_matches_sample_df['match_cluster_id'],
        categories=all_possible_cluster_ids
    )
    cluster_dummies_new = pd.get_dummies(new_matches_sample_df['match_cluster_id'], prefix='cluster', dtype=int)

    X_new = pd.concat([X_new, cluster_dummies_new], axis=1)
    print(f"Added {cluster_dummies_new.shape[1]} cluster dummy features to X_new. Shape of X_new after cluster features: {X_new.shape}")

    # One-Hot Encode 'RelativeUnderdog' for the sample, if present in historical_df
    if 'RelativeUnderdog' in historical_df.columns:
        if 'RelativeUnderdog' in new_matches_sample_df.columns: # Check if also in sample, though it should be if in historical_df
            print("Applying One-Hot Encoding to 'RelativeUnderdog' for X_new...")
            all_possible_underdog_states = sorted(historical_df['RelativeUnderdog'].unique())
            new_matches_sample_df['RelativeUnderdog'] = pd.Categorical(
                new_matches_sample_df['RelativeUnderdog'],
                categories=all_possible_underdog_states
            )
            underdog_dummies_new = pd.get_dummies(new_matches_sample_df['RelativeUnderdog'], prefix='Underdog', dtype=int)
            X_new = pd.concat([X_new, underdog_dummies_new], axis=1)
            print(f"Added {underdog_dummies_new.shape[1]} RelativeUnderdog dummy features to X_new. Shape of X_new after Underdog features: {X_new.shape}")
        else:
            # This case should ideally not happen if RelativeUnderdog is in historical_df and new_matches_sample_df is a slice
            print("Warning: 'RelativeUnderdog' found in historical_df but not in new_matches_sample_df. Omitting Underdog features from X_new.")
            # To ensure no missing columns if model expects them, create all possible underdog dummies with 0s.
            # However, model_trainer.py also has a conditional addition, so if not present, model wasn't trained with them.
            # This is aligned with model_trainer.py's behavior (trains without it if column is missing there).
    else:
        print("Warning: 'RelativeUnderdog' column not found in historical_df. Proceeding without this feature in X_new.")

    # Ensure X_new does not have NaNs that might have been introduced by pd.Categorical
    # or if original rolling features had NaNs not caught by earlier dropna.
    if X_new.isnull().sum().any().any():
        print("Warning: NaNs detected in X_new before prediction. This is unexpected. Filling with 0.")
        X_new.fillna(0, inplace=True)

    # At this point, X_new should have the same columns as X_train in model_trainer.py
    # A truly robust solution would involve loading the expected columns from training.
    # For now, we assume the model object handles feature names or order, or they align by construction.

    # 5. Make Predictions
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    # 6. Display Predictions
    print("\n--- Predictions for Simulated New Matches ---")
    # Iterate using index from new_matches_sample_df to ensure correct row alignment
    for i, index_val in enumerate(new_matches_sample_df.index): # Iterate over the sample's index
        match_details = new_matches_sample_df.loc[index_val] # Use .loc with the actual index from the sample

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
