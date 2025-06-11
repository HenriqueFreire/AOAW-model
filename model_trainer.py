import os
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
# from collections import deque # No longer needed as get_rolling_features is removed

# --- Load Preprocessed and Clustered Data ---
# This file is expected to be created by cluster_features.py
# It should contain rolling features and 'match_cluster_id'.
try:
    # Renaming to df_featured as it already contains features
    df_featured = pd.read_csv('/app/processed_data_clustered.csv', parse_dates=['Date'])
    print(f"Successfully loaded /app/processed_data_clustered.csv. Shape: {df_featured.shape}")
    # Sorting by date might be good practice, though not strictly needed if features are pre-calculated
    df_featured.sort_values(by='Date', inplace=True)
except FileNotFoundError:
    print("Error: /app/processed_data_clustered.csv not found. Please ensure cluster_features.py has run successfully.")
    exit()
except Exception as e:
    print(f"Error loading /app/processed_data_clustered.csv: {e}")
    exit()

# Helper functions get_match_points and get_rolling_features are removed as features are pre-calculated.

# --- Main Script Logic ---
print("Starting model trainer script using clustered data...")

# Section 1 & 2 (Data Downloading, Initial Preprocessing) and
# Section 3 (Feature Engineering with get_rolling_features) are effectively replaced by loading processed_data_clustered.csv

# 3. Data Preparation (Post-Feature Engineering)
print("\n--- Section 3: Preparing Data for Model Training ---")
if df_featured.empty:
    print("Loaded dataframe from /app/processed_data_clustered.csv is empty. Cannot proceed.")
    exit()

# Drop NaNs - this is a safety measure.
# cluster_features.py should have already handled NaNs from rolling features.
# This might handle NaNs in other columns (e.g. odds if they were not used as features but are present).
df_featured.dropna(inplace=True)
print(f"Shape after final NaN drop (if any): {df_featured.shape}")

if df_featured.empty:
    print("DataFrame is empty after NaN drop. Cannot train model.")
    exit()

# Check if 'match_cluster_id' is present
if 'match_cluster_id' not in df_featured.columns:
    print("Error: 'match_cluster_id' column not found in the loaded data. Ensure cluster_features.py ran correctly.")
    exit()

# 4. Model Training
print("\n--- Section 4: Model Training with Cluster Features ---")

# Define base rolling features
rolling_feature_cols = [
    'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
    'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded'
]
target_col = 'LayAOAV'

# Verify all rolling feature columns are present
missing_rolling_features = [col for col in rolling_feature_cols if col not in df_featured.columns]
if missing_rolling_features:
    print(f"Error: Missing rolling feature columns: {missing_rolling_features}")
    exit()

if target_col not in df_featured.columns:
    print(f"Error: Target column '{target_col}' not found.")
    exit()

# Create initial X with rolling features
X = df_featured[rolling_feature_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

# One-Hot Encode 'match_cluster_id'
print("Applying One-Hot Encoding to 'match_cluster_id'...")
cluster_dummies = pd.get_dummies(df_featured['match_cluster_id'], prefix='cluster', dtype=int)
X = pd.concat([X, cluster_dummies], axis=1)
print(f"Added {cluster_dummies.shape[1]} cluster dummy features. Shape of X after cluster features: {X.shape}")

# One-Hot Encode 'RelativeUnderdog' if present
if 'RelativeUnderdog' in df_featured.columns:
    print("Applying One-Hot Encoding to 'RelativeUnderdog'...")
    underdog_dummies = pd.get_dummies(df_featured['RelativeUnderdog'], prefix='Underdog', dtype=int)
    X = pd.concat([X, underdog_dummies], axis=1)
    print(f"Added {underdog_dummies.shape[1]} RelativeUnderdog dummy features. Shape of X after Underdog features: {X.shape}")
else:
    print("Warning: 'RelativeUnderdog' column not found. Proceeding without this feature.")

y = df_featured[target_col].astype(int)

# The full list of feature columns is now X.columns
# This check is now more robust as it uses the actual columns in X
if not all(col in X.columns for col in X.columns): # This will always be true, but serves as placeholder
    # This check could be: if X.empty or len(X.columns) == len(rolling_feature_cols) if cluster_dummies.shape[1] == 0
    # Meaning no cluster features were added, which could be an issue.
    # For now, we assume get_dummies works as expected if 'match_cluster_id' exists.
    pass


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
model_filename = '/app/model_lay_aoav.joblib' # Saving in /app for persistence if environment allows
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

print("\nConsolidated model trainer script finished.")
