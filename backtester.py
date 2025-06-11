import pandas as pd
import numpy as np
import joblib
# from collections import deque # No longer needed
import os

# Helper functions get_match_points and get_rolling_features are removed
# as features are expected to be pre-calculated in the input CSV.

def run_backtesting():
    print("--- Starting Backtester Script (using clustered data) ---")

    # 1. Load Artifacts
    model_path = '/app/model_lay_aoav.joblib'
    # Load data from the clustered CSV
    data_path = '/app/processed_data_clustered.csv'

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return
    if not os.path.exists(data_path):
        print(f"Error: Clustered data file not found at {data_path}. Please ensure 'cluster_features.py' has run.")
        return

    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        # Load the full dataset which already contains features and cluster IDs
        # Renaming to full_data_df as it contains all data, not just features.
        full_data_df = pd.read_csv(data_path, parse_dates=['Date'])
        full_data_df.sort_values(by='Date', inplace=True)
        print(f"Clustered data loaded from {data_path}. Shape: {full_data_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if full_data_df.empty:
        print("Loaded clustered data is empty. Exiting.")
        return

    # Ensure LayAOAV (target) and cluster_id are present
    if 'LayAOAV' not in full_data_df.columns:
        print(f"Error: Target column 'LayAOAV' not found in the loaded data.")
        return
    if 'match_cluster_id' not in full_data_df.columns:
        print(f"Error: 'match_cluster_id' column not found. Ensure cluster_features.py ran correctly.")
        return

    # 2. Data Preparation (Post-Feature Engineering from loaded file)
    # Rolling features are already in full_data_df.
    # Apply dropna as a safety measure. cluster_features.py should have handled NaNs
    # for rolling features, but this handles any other potential NaNs.
    full_data_df.dropna(inplace=True)
    print(f"Shape of full_data_df after NaN drop: {full_data_df.shape}")

    if full_data_df.empty:
        print("DataFrame is empty after NaN drop. Cannot proceed.")
        return

    # 3. Data Preparation for Backtesting
    rolling_feature_cols = [
        'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
        'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded'
    ]
    target_col = 'LayAOAV'

    # Verify all rolling feature columns are present
    missing_rolling_features = [col for col in rolling_feature_cols if col not in full_data_df.columns]
    if missing_rolling_features:
        print(f"Error: Missing rolling feature columns in loaded data: {missing_rolling_features}")
        return

    # Chronological Split: Use last 20% for backtesting
    backtest_size = int(len(full_data_df) * 0.20)
    if backtest_size == 0 and len(full_data_df) > 0:
        backtest_size = min(len(full_data_df), 5)

    if backtest_size == 0:
        print("Not enough data to create a backtesting set. Exiting.")
        return

    backtest_df = full_data_df.iloc[-backtest_size:].copy()

    # Prepare X_backtest with rolling features and one-hot encoded cluster IDs
    X_backtest = backtest_df[rolling_feature_cols].copy()

    # One-Hot Encode 'match_cluster_id' for the backtest set
    # Use categories from the full loaded dataset to ensure consistency
    all_possible_cluster_ids = sorted(full_data_df['match_cluster_id'].unique())
    backtest_df['match_cluster_id'] = pd.Categorical(
        backtest_df['match_cluster_id'],
        categories=all_possible_cluster_ids
    )
    cluster_dummies_backtest = pd.get_dummies(backtest_df['match_cluster_id'], prefix='cluster', dtype=int)

    X_backtest = pd.concat([X_backtest, cluster_dummies_backtest], axis=1)
    print(f"Added {cluster_dummies_backtest.shape[1]} cluster dummy features to X_backtest. Shape of X_backtest after cluster features: {X_backtest.shape}")

    # One-Hot Encode 'RelativeUnderdog' for the backtest set, if present in full_data_df
    if 'RelativeUnderdog' in full_data_df.columns:
        if 'RelativeUnderdog' in backtest_df.columns: # Should be true if in full_data_df
            print("Applying One-Hot Encoding to 'RelativeUnderdog' for X_backtest...")
            all_possible_underdog_states = sorted(full_data_df['RelativeUnderdog'].unique())
            backtest_df['RelativeUnderdog'] = pd.Categorical(
                backtest_df['RelativeUnderdog'],
                categories=all_possible_underdog_states
            )
            underdog_dummies_backtest = pd.get_dummies(backtest_df['RelativeUnderdog'], prefix='Underdog', dtype=int)
            X_backtest = pd.concat([X_backtest, underdog_dummies_backtest], axis=1)
            print(f"Added {underdog_dummies_backtest.shape[1]} RelativeUnderdog dummy features to X_backtest. Shape of X_backtest after Underdog features: {X_backtest.shape}")
        else:
            print("Warning: 'RelativeUnderdog' found in full_data_df but not in backtest_df. Omitting Underdog features from X_backtest.")
    else:
        print("Warning: 'RelativeUnderdog' column not found in full_data_df. Proceeding without this feature in X_backtest.")

    y_backtest_actual = backtest_df[target_col].astype(int)

    print(f"Backtesting set size: {len(X_backtest)} matches.")

    if X_backtest.empty:
        print("Backtesting feature set (X_backtest) is empty. Cannot proceed.")
        return

    # 4. Strategy Simulation & Evaluation
    probability_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    print(f"\n--- Backtesting LayAOAV Strategy with various thresholds ---")
    print("Strategy: Bet 'Lay AOAV' (i.e., predict LayAOAV=1) if P(LayAOAV=1) > threshold.")

    # Get probabilities for P(LayAOAV=1)
    # model.classes_ gives the order of classes; find index of class 1
    try:
        class_1_index = model.classes_.tolist().index(1)
    except ValueError:
        print("Error: Class '1' not found in model.classes_. This model might not be for LayAOAV.")
        # If only one class is present in model.classes_ (e.g. only 0s or only 1s in y_train)
        # or if classes are not [0, 1], this could be an issue.
        # For a binary classifier for LayAOAV, we expect classes [0, 1].
        if len(model.classes_) == 1 and model.classes_[0] == 0:
             print("Model only predicts class 0. No 'Lay' bets will be placed.")
             # Set class_1_index to an invalid index or handle so P(LayAOAV=1) is effectively 0
             # This scenario means the model never predicts LayAOAV=1.
             # For this specific strategy, no bets would be made.
             class_1_index = -1 # Sentinel to indicate P(LayAOAV=1) is effectively 0
        elif len(model.classes_) == 1 and model.classes_[0] == 1:
             print("Model only predicts class 1. All bets (if prob > threshold) would be 'Lay'.")
             # This is also an extreme scenario.
             class_1_index = 0 # if model.classes_ is [1], then proba is for class 1
        else:
            print(f"Model classes are: {model.classes_}. Cannot determine P(LayAOAV=1).")
            return


    y_pred_proba_backtest = model.predict_proba(X_backtest)

    results_summary = []

    for threshold in probability_thresholds:
        simulated_bets_count = 0
        successful_bets_count = 0

        for i in range(len(X_backtest)):
            # Probability of LayAOAV=1 (Away team does NOT win by 4+ goals)
            prob_lay_aoav_is_1 = 0
            if class_1_index != -1 : # if class 1 is a possible output
                 prob_lay_aoav_is_1 = y_pred_proba_backtest[i, class_1_index]

            if prob_lay_aoav_is_1 > threshold:
                simulated_bets_count += 1
                actual_outcome = y_backtest_actual.iloc[i]
                if actual_outcome == 1: # Lay bet won
                    successful_bets_count += 1

        hit_rate = 0
        if simulated_bets_count > 0:
            hit_rate = successful_bets_count / simulated_bets_count

        results_summary.append({
            "Threshold": threshold,
            "Simulated Bets": simulated_bets_count,
            "Successful Bets": successful_bets_count,
            "Hit Rate": f"{hit_rate:.2%}"
        })
        print(f"Threshold: {threshold:.2f} | Bets: {simulated_bets_count:4} | Successful: {successful_bets_count:4} | Hit Rate: {hit_rate:.2%}")

    print("\n--- Backtesting Summary ---")
    for res in results_summary:
        print(f"Threshold: {res['Threshold']:.2f}, Bets Placed: {res['Simulated Bets']}, Successful Bets: {res['Successful Bets']}, Hit Rate: {res['Hit Rate']}")

    print("\n--- Backtester Script Finished ---")

if __name__ == '__main__':
    run_backtesting()
