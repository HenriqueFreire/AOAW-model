import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression

print("Starting model trainer script for Asian Handicap (AH) - 'Lay the Zebra' strategy (with Rolling Features)...")

# 1. Carregar Dados Processados para AH (que agora incluem rolling features)
print("\n--- Section 1: Loading Processed AH Data (with Rolling Features) ---")
processed_data_path = './app/processed_data_ah.csv'
if not os.path.exists(processed_data_path):
    print(f"Error: Processed AH data file not found at {processed_data_path}. Please run data_processor.py first.")
    exit()

try:
    df_processed = pd.read_csv(processed_data_path)
    print(f"Successfully loaded processed AH data from {processed_data_path}. Shape: {df_processed.shape}")
except Exception as e:
    print(f"Error loading processed AH data: {e}")
    exit()

if df_processed.empty:
    print("Processed AH data is empty. Cannot train model.")
    exit()

# Nomes das features de rolling average como geradas pelo data_processor.py
rolling_feature_names = [
    'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded', 'HomeTeam_RecentForm',
    'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded', 'AwayTeam_RecentForm'
]

# Features base (Odds de Match e AH)
base_feature_cols = [
    'B365H', 'B365D', 'B365A',
    'PSCH', 'PSCA',
    'PCAHH', 'PCAHA'
]

# Todas as features que o modelo usará
all_feature_cols_for_model = base_feature_cols + rolling_feature_names

# Colunas requeridas no arquivo CSV (incluindo Season e Target)
required_cols_for_training = ['Season'] + all_feature_cols_for_model + ['ZebraLosesOrDraws']

missing_cols = [col for col in required_cols_for_training if col not in df_processed.columns]
if missing_cols:
    print(f"Error: Missing required columns in processed AH data: {missing_cols}. Ensure data_processor.py ran correctly and includes rolling features.")
    exit()

# Converter colunas de features para numérico, tratando erros como NaN
# Rolling features já devem ser numéricas ou NaN a partir do data_processor
feature_cols_to_convert_numeric = base_feature_cols # Rolling features já são float ou NaN

for col in feature_cols_to_convert_numeric:
    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

# Remover linhas com NaN em QUALQUER UMA das features que o modelo usará, ou na Season/Target.
# As rolling features terão NaNs nos primeiros jogos, que serão removidos aqui.
cols_to_check_na = ['Season'] + all_feature_cols_for_model + ['ZebraLosesOrDraws']
df_processed.dropna(subset=cols_to_check_na, inplace=True)
print(f"Shape after dropping NaNs from all model features, Season, and Target: {df_processed.shape}")

if df_processed.empty:
    print("Dataframe is empty after NaN removal. Cannot train model.")
    exit()

# 2. Selecionar Dados de Treinamento (Duas Primeiras Temporadas)
print("\n--- Section 2: Selecting Training Data (First Two Seasons) for AH Model ---")
df_processed['Season'] = pd.to_numeric(df_processed['Season'], errors='coerce').astype(int)

unique_seasons = sorted(df_processed['Season'].unique())
if len(unique_seasons) < 2:
    print(f"Error: Less than two unique seasons available for AH training ({len(unique_seasons)} found).")
    exit()

training_seasons = unique_seasons[:2]
print(f"Training AH model using data from seasons: {training_seasons}")

df_train = df_processed[df_processed['Season'].isin(training_seasons)]
print(f"Shape of AH training data (from first two seasons): {df_train.shape}")

if df_train.empty:
    print("No data available for AH training after selecting the first two seasons.")
    exit()

target_col_ah = 'ZebraLosesOrDraws'
if len(df_train[target_col_ah].unique()) < 2:
    print(f"Warning: Only one class present in {target_col_ah} for the training seasons. Model may not train well. Value counts: {df_train[target_col_ah].value_counts()}")
    if len(df_train) < 20:
        print("AH training data too small or lacks class diversity. Exiting.")
        exit()

# 3. Treinamento do Modelo AH (com Rolling Features)
print("\n--- Section 3: AH Model Training (with Rolling Features) ---")
X_train = df_train[all_feature_cols_for_model] # Usando todas as features definidas
y_train = df_train[target_col_ah].astype(int)

model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
try:
    model.fit(X_train, y_train)
    print(f"AH Model ({target_col_ah}) trained successfully using features: {all_feature_cols_for_model}.")
except ValueError as ve:
    print(f"Error during AH model training: {ve}.")
    exit()

# 4. Salvar Modelo AH
print("\n--- Section 4: Saving AH Model ---")
output_dir = './app/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

model_filename = os.path.join(output_dir, 'model_asian_handicap.joblib')
try:
    joblib.dump(model, model_filename)
    print(f"AH Model (with rolling features) saved to {model_filename}")
except Exception as e:
    print(f"Error saving AH model: {e}")
    exit()

print("\nAH Model trainer script (with rolling features) finished.")
```
