import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import deque
import os

# --- Helper Functions (copied from model_trainer.py / backtester.py) ---
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

def get_rolling_features(df, N=5):
    df_copy = df.sort_values(by='Date').copy() # Ensure sorting by Date

    home_team_stats = {}
    away_team_stats = {}

    # Define feature names within the function context as it's self-contained
    rolling_feature_names = [
        'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
        'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded'
    ]
    for col in rolling_feature_names:
        df_copy[col] = np.nan

    all_teams = pd.concat([df_copy['HomeTeam'], df_copy['AwayTeam']]).unique()
    for team in all_teams:
        home_team_stats[team] = deque(maxlen=N)
        away_team_stats[team] = deque(maxlen=N)

    for index, row in df_copy.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Calculate for Home Team (must have N previous games in its deque)
        if len(home_team_stats[home_team]) == N:
            past_N_games = list(home_team_stats[home_team])
            df_copy.loc[index, 'HomeTeam_RecentPoints'] = sum(g['points'] for g in past_N_games)
            df_copy.loc[index, 'HomeTeam_RecentGoalsScored'] = sum(g['goals_scored'] for g in past_N_games)
            df_copy.loc[index, 'HomeTeam_RecentGoalsConceded'] = sum(g['goals_conceded'] for g in past_N_games)

        # Calculate for Away Team (must have N previous games in its deque)
        if len(away_team_stats[away_team]) == N:
            past_N_games = list(away_team_stats[away_team])
            df_copy.loc[index, 'AwayTeam_RecentPoints'] = sum(g['points'] for g in past_N_games)
            df_copy.loc[index, 'AwayTeam_RecentGoalsScored'] = sum(g['goals_scored'] for g in past_N_games)
            df_copy.loc[index, 'AwayTeam_RecentGoalsConceded'] = sum(g['goals_conceded'] for g in past_N_games)

        # Append current game stats for future calculations
        home_points = get_match_points(row['FTR'], 'home')
        home_team_stats[home_team].append({'points': home_points, 'goals_scored': row['FTHG'], 'goals_conceded': row['FTAG']})

        away_points = get_match_points(row['FTR'], 'away')
        away_team_stats[away_team].append({'points': away_points, 'goals_scored': row['FTAG'], 'goals_conceded': row['FTHG']})

    return df_copy

def main():
    print("--- Starting Feature Clustering Script ---")

    # 1. Load Data
    data_path = '/app/processed_data_lay_aoav.csv'
    output_path = '/app/processed_data_clustered.csv'

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please ensure 'data_processor.py' has run.")
        return

    try:
        df = pd.read_csv(data_path, parse_dates=['Date'])
        df.sort_values(by='Date', inplace=True) # Essential for get_rolling_features
        print(f"Data loaded successfully from {data_path}. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print("Loaded dataframe is empty. Exiting.")
        return

    # 2. Generate Rolling Features
    print("Generating rolling features...")
    # Ensure all necessary columns for get_rolling_features are present
    required_cols_for_rolling_features = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    missing_cols = [col for col in required_cols_for_rolling_features if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns required for get_rolling_features: {missing_cols}. Exiting.")
        return

    df_featured = get_rolling_features(df.copy(), N=5) # Use .copy() to avoid modifying original df

    # Remove rows with NaN values (typically earliest matches without enough historical data)
    df_featured.dropna(inplace=True)
    print(f"Rolling features generated. Shape after NaN drop: {df_featured.shape}")

    if df_featured.empty:
        print("Dataframe is empty after generating rolling features and dropping NaNs. Exiting.")
        return

    # 3. Select Features for Clustering
    features_for_clustering_cols = [
        'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
        'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded'
    ]

    # Verify that these columns exist in df_featured (they should, if get_rolling_features worked)
    missing_cluster_features = [col for col in features_for_clustering_cols if col not in df_featured.columns]
    if missing_cluster_features:
        print(f"Error: One or more features intended for clustering are missing: {missing_cluster_features}. This is unexpected. Exiting.")
        return

    X_for_clustering = df_featured[features_for_clustering_cols]
    print(f"Selected features for clustering. Shape: {X_for_clustering.shape}")

    # 4. Normalize Features
    print("Normalizing features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_for_clustering)
    print("Features normalized.")

    # 5. Apply K-means Clustering
    k = 4
    print(f"Applying K-means clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    print("K-means clustering applied.")

    # 6. Add Cluster IDs to DataFrame
    cluster_labels = kmeans.labels_
    # Ensure that the labels are added to the correct DataFrame.
    # df_featured at this point has already been through dropna, and X_for_clustering was derived from it.
    # So, the indices should align.
    df_featured['match_cluster_id'] = cluster_labels
    print(f"Cluster labels added to DataFrame as 'match_cluster_id'.")

    # 7. Save DataFrame Resultante
    try:
        df_featured.to_csv(output_path, index=False)
        print(f"Clustered data saved successfully to {output_path}")
        print("First few rows of the clustered data:")
        print(df_featured.head())
        print(f"Value counts for 'match_cluster_id':")
        print(df_featured['match_cluster_id'].value_counts())

    except Exception as e:
        print(f"Error saving clustered data: {e}")

    print("--- Feature Clustering Script Finished ---")

if __name__ == '__main__':
    main()
