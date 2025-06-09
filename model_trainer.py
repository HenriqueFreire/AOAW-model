import pandas as pd
import numpy as np
import joblib
import os # Adicionado para os caminhos de arquivo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
# A função get_rolling_features e seu uso serão removidos nesta versão para focar nas odds.
# A lógica de download e pré-processamento também será removida.

print("Starting model trainer script for backtesting...")

# 1. Carregar Dados Processados
print("
--- Section 1: Loading Processed Data ---")
processed_data_path = './app/processed_data_lay_aoav.csv'
if not os.path.exists(processed_data_path):
    print(f"Error: Processed data file not found at {processed_data_path}. Please run data_processor.py first.")
    exit()

try:
    df_processed = pd.read_csv(processed_data_path)
    print(f"Successfully loaded processed data from {processed_data_path}. Shape: {df_processed.shape}")
except Exception as e:
    print(f"Error loading processed data: {e}")
    exit()

if df_processed.empty:
    print("Processed data is empty. Cannot train model.")
    exit()

# Verificar se a coluna 'Season' e as colunas de odds existem
required_cols_for_training = ['Season', 'B365H', 'B365D', 'B365A', 'LayAOAV']
missing_cols = [col for col in required_cols_for_training if col not in df_processed.columns]
if missing_cols:
    print(f"Error: Missing required columns in processed data: {missing_cols}. Ensure data_processor.py ran correctly.")
    exit()

# Converter colunas de odds para numérico, tratando erros como NaN (que devem ser removidos)
for col in ['B365H', 'B365D', 'B365A']:
    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

# Remover linhas com NaN nas features ou target que usaremos
cols_to_check_na = ['Season', 'B365H', 'B365D', 'B365A', 'LayAOAV']
df_processed.dropna(subset=cols_to_check_na, inplace=True)
print(f"Shape after dropping NaNs from essential columns for training: {df_processed.shape}")

if df_processed.empty:
    print("Dataframe is empty after NaN removal from essential columns. Cannot train model.")
    exit()


# 2. Selecionar Dados de Treinamento (Duas Primeiras Temporadas)
print("
--- Section 2: Selecting Training Data (First Two Seasons) ---")
# Assegurar que 'Season' é numérica para ordenação correta
df_processed['Season'] = pd.to_numeric(df_processed['Season'], errors='coerce')
df_processed.dropna(subset=['Season'], inplace=True) # Remover se a conversão falhar
df_processed['Season'] = df_processed['Season'].astype(int)

unique_seasons = sorted(df_processed['Season'].unique())
if len(unique_seasons) < 2:
    print(f"Error: Less than two unique seasons available in the data ({len(unique_seasons)} found). Cannot select two seasons for training.")
    exit()

training_seasons = unique_seasons[:2]
print(f"Training model using data from seasons: {training_seasons}")

df_train = df_processed[df_processed['Season'].isin(training_seasons)]
print(f"Shape of training data (from first two seasons): {df_train.shape}")

if df_train.empty:
    print("No data available for training after selecting the first two seasons.")
    exit()
if len(df_train['LayAOAV'].unique()) < 2:
    print(f"Warning: Only one class present in LayAOAV for the training seasons. Model may not train well. Value counts: {df_train['LayAOAV'].value_counts()}")
    if len(df_train) < 10: # Arbitrariamente pequeno para não treinar
        print("Training data too small or lacks class diversity. Exiting.")
        exit()

# 3. Treinamento do Modelo
print("
--- Section 3: Model Training ---")
feature_cols = ['B365H', 'B365D', 'B365A'] # Usando apenas odds de Match Odds como features
target_col = 'LayAOAV'

X_train = df_train[feature_cols]
y_train = df_train[target_col].astype(int)

# Nota: Não há divisão X_test, y_test aqui, pois o backtester.py fará a avaliação em dados futuros.
# Treinamos com todos os dados selecionados (das duas primeiras temporadas).
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
try:
    model.fit(X_train, y_train)
    print("Model trained successfully using data from the first two seasons.")
except ValueError as ve:
    print(f"Error during model training: {ve}. This might happen if only one class is present in y_train.")
    exit()


# 4. Salvar Modelo para Backtesting
print("
--- Section 4: Saving Model for Backtesting ---")
# Cria o diretório ./app/ se não existir
output_dir = './app/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

model_filename = os.path.join(output_dir, 'model_lay_aoav_backtest.joblib')
try:
    joblib.dump(model, model_filename)
    print(f"Model for backtesting saved to {model_filename}")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

print("
Model trainer script for backtesting finished.")

```
