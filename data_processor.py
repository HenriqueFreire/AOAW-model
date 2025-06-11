import pandas as pd
import numpy as np
import os
import glob

# --- Funções Auxiliares Removidas (get_match_points, get_rolling_features) ---

FILE_PATTERN = "./app/E0_*.csv"
dataframes = []

print("Starting data processing for 'Lay the Zebra' strategy (Features: Match Odds Only)...")

csv_files = sorted(glob.glob(FILE_PATTERN))

if not csv_files:
    print(f"No CSV files found in ./app/ matching pattern {FILE_PATTERN}. Ensure downloader.py has run successfully.")
    exit()
else:
    print(f"Found the following CSV files to process: {csv_files}")
    for filename in csv_files:
        print(f"Processing file: {filename}")
        try:
            df = pd.read_csv(filename, encoding='ISO-8859-1')
            base_name = os.path.basename(filename)
            season_code_str = base_name.replace('E0_', '').replace('.csv', '')
            season_start_year = pd.NA
            if season_code_str == "2021": # Temporada 2020/2021
                season_start_year = 2020
            elif len(season_code_str) == 4 and season_code_str.isdigit(): # Ex: 2122, 2223, 2324
                season_start_year = 2000 + int(season_code_str[:2])
            else:
                print(f"Warning: Could not determine season year for {filename} (code: {season_code_str}).")
            df['Season'] = season_start_year
            dataframes.append(df)
        except Exception as e:
            print(f"Warning: Could not read or process {filename} due to error: {e}")

if not dataframes:
    print("No dataframes were loaded. Exiting.")
    exit()

combined_df = pd.concat(dataframes, ignore_index=True)
print(f"Combined dataframes. Total rows before processing: {len(combined_df)}")

if combined_df.empty:
    print("Combined dataframe is empty. No data to process.")
    exit()

# --- Início da Lógica Específica para "Lay à Zebra" (Features Simplificadas) ---

# 0. Limpeza Inicial e Ordenação (Removida referência a rolling features)
print("Performing initial sort and data type conversions...")
if 'Date' in combined_df.columns:
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce', dayfirst=True)
else:
    print("Critical Error: 'Date' column missing. Cannot proceed.")
    exit()

if 'Season' in combined_df.columns:
    combined_df['Season'] = pd.to_numeric(combined_df['Season'], errors='coerce')
else:
    print("Critical Error: 'Season' column missing. Cannot proceed.")
    exit()

for col in ['FTHG', 'FTAG']: # Necessários para a variável alvo
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    else:
        print(f"Critical Error: Column '{col}' missing.")
        exit()
if 'FTR' not in combined_df.columns: # Necessário para a variável alvo
    print("Critical Error: 'FTR' column missing.")
    exit()

initial_essential_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
combined_df.dropna(subset=initial_essential_cols, inplace=True)
combined_df.sort_values(by=['Season', 'Date'], inplace=True)
print(f"Rows after initial essential NaN drop and sort: {len(combined_df)}")

# --- Seção de Cálculo de Rolling Features Removida ---

# 1. Selecionar Colunas Essenciais de Odds (Apenas Match Odds)
# Usaremos B365 para Match Odds como padrão.
match_odds_cols_map = {
    'B365H': 'B365H', 'B365D': 'B365D', 'B365A': 'B365A'
}
for col_original, col_new in match_odds_cols_map.items():
    if col_original not in combined_df.columns:
        print(f"Warning: Match Odds column {col_original} not found. Will be NaN.")
        combined_df[col_new] = np.nan # Garante que a coluna exista
    elif col_original != col_new: # Se precisasse renomear
        combined_df[col_new] = combined_df[col_original]

# Colunas a manter: Base + Match Odds
cols_to_keep_base = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
cols_to_keep_match_odds = ['B365H', 'B365D', 'B365A']
# Colunas de AH e Rolling Features foram removidas da seleção

# Lista final de colunas antes de adicionar Zebra e ZebraLosesOrDraws
cols_for_processing = cols_to_keep_base + cols_to_keep_match_odds
current_existing_cols = [col for col in cols_for_processing if col in combined_df.columns]
combined_df = combined_df[current_existing_cols]

# 2. Conversão de Tipos para Match Odds
for col in cols_to_keep_match_odds:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    else: # Deveria ter sido criada com NaN se não existia
        combined_df[col] = np.nan


# 3. Identificar a Zebra (usando Match Odds)
critical_for_zebra_id = ['B365H', 'B365D', 'B365A', 'FTR'] # FTR é usado para a target
print(f"Rows before dropping NaNs for Zebra ID (Match Odds): {len(combined_df)}")
combined_df.dropna(subset=critical_for_zebra_id, inplace=True)
print(f"Rows after dropping NaNs for Zebra ID (Match Odds): {len(combined_df)}")

combined_df['Zebra'] = np.where(combined_df['B365H'] > combined_df['B365A'], 'H', 'A')
combined_df['Zebra'] = np.where(combined_df['B365H'] == combined_df['B365A'], 'A', combined_df['Zebra'])

# 4. Criar Variável Alvo: ZebraLosesOrDraws
conditions_zebra_loses_or_draws = [
    (combined_df['Zebra'] == 'H') & (combined_df['FTR'].isin(['A', 'D'])),
    (combined_df['Zebra'] == 'A') & (combined_df['FTR'].isin(['H', 'D']))
]
choices = [1, 1]
combined_df['ZebraLosesOrDraws'] = np.select(conditions_zebra_loses_or_draws, choices, default=0)

# 5. Tratamento Final de NaNs (Agora apenas para Match Odds, se ainda houver)
# A etapa critical_for_zebra_id já deve ter tratado NaNs nas Match Odds.
# Esta etapa é mais uma garantia ou se outras features fossem adicionadas.
final_feature_cols_to_check_na = cols_to_keep_match_odds
final_feature_cols_to_check_na_existing = [col for col in final_feature_cols_to_check_na if col in combined_df.columns]

if final_feature_cols_to_check_na_existing:
    print(f"Rows before final feature NaN drop: {len(combined_df)}")
    combined_df.dropna(subset=final_feature_cols_to_check_na_existing, inplace=True)
    print(f"Rows after final feature NaN drop: {len(combined_df)}")

# 6. Conversões Finais de Tipo e Seleção Final de Colunas
if 'FTHG' in combined_df.columns: combined_df['FTHG'] = combined_df['FTHG'].astype(int)
if 'FTAG' in combined_df.columns: combined_df['FTAG'] = combined_df['FTAG'].astype(int)
if 'Season' in combined_df.columns: combined_df['Season'] = combined_df['Season'].astype(int)

# Colunas finais a serem salvas
final_selected_cols = cols_to_keep_base + cols_to_keep_match_odds + ['Zebra', 'ZebraLosesOrDraws']
final_selected_cols_existing = [col for col in final_selected_cols if col in combined_df.columns]
combined_df = combined_df[final_selected_cols_existing]

# --- Fim da Lógica Específica para "Lay à Zebra" ---

output_filepath = './app/processed_data_ah.csv' # Mantém o mesmo nome de output por enquanto
combined_df.to_csv(output_filepath, index=False)

print(f"Data processed for AH 'Lay the Zebra' strategy (Features: Match Odds Only) and saved to {output_filepath}")
print("Head of the processed data:")
print(combined_df.head())
if 'ZebraLosesOrDraws' in combined_df.columns:
    print("Value counts for 'ZebraLosesOrDraws':")
    print(combined_df['ZebraLosesOrDraws'].value_counts(normalize=True))
print(f"Shape of processed data: {combined_df.shape}")

print("Data processing script for Asian Handicap (AH) - 'Lay the Zebra' (Features: Match Odds Only) finished.")

```
