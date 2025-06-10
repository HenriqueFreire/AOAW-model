import pandas as pd
import numpy as np
import os
import glob
from collections import deque # Importar deque para get_rolling_features

# --- Início das Funções Auxiliares ---
def get_match_points(result_indicator, perspective):
    # Função auxiliar para get_rolling_features
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
    """Calcula features de rolling average para os últimos N jogos de cada time."""
    df_copy = df.copy() # Trabalhar em uma cópia para evitar SettingWithCopyWarning

    # Certificar que FTHG, FTAG, FTR existem e são dos tipos corretos antes de usá-los.
    # Esta função espera que o DataFrame df_copy já tenha passado por uma limpeza inicial
    # e que 'Date' esteja em formato datetime.
    # A ordenação por Date (e Season) deve ser feita ANTES de chamar esta função.

    home_team_stats = {}
    away_team_stats = {}

    # Nomes das novas features de rolling
    rolling_feature_names = [
        'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded',
        'HomeTeam_RecentForm', # Média de pontos nos últimos N jogos
        'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded',
        'AwayTeam_RecentForm'  # Média de pontos nos últimos N jogos
    ]
    for col in rolling_feature_names:
        df_copy[col] = np.nan

    all_teams = pd.concat([df_copy['HomeTeam'], df_copy['AwayTeam']]).unique()
    for team in all_teams:
        home_team_stats[team] = deque(maxlen=N)
        # Away stats também usa a perspectiva do time (seus gols marcados/sofridos)
        away_team_stats[team] = deque(maxlen=N)


    # Iterar sobre o DataFrame ordenado por data para calcular as features progressivamente
    for index, row in df_copy.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Home Team Rolling Features
        if len(home_team_stats[home_team]) == N:
            past_N_games_home = list(home_team_stats[home_team])
            df_copy.loc[index, 'HomeTeam_RecentPoints'] = sum(g['points'] for g in past_N_games_home)
            df_copy.loc[index, 'HomeTeam_RecentGoalsScored'] = sum(g['goals_scored'] for g in past_N_games_home)
            df_copy.loc[index, 'HomeTeam_RecentGoalsConceded'] = sum(g['goals_conceded'] for g in past_N_games_home)
            df_copy.loc[index, 'HomeTeam_RecentForm'] = sum(g['points'] for g in past_N_games_home) / N


        # Away Team Rolling Features
        if len(away_team_stats[away_team]) == N:
            past_N_games_away = list(away_team_stats[away_team])
            df_copy.loc[index, 'AwayTeam_RecentPoints'] = sum(g['points'] for g in past_N_games_away)
            df_copy.loc[index, 'AwayTeam_RecentGoalsScored'] = sum(g['goals_scored'] for g in past_N_games_away)
            df_copy.loc[index, 'AwayTeam_RecentGoalsConceded'] = sum(g['goals_conceded'] for g in past_N_games_away)
            df_copy.loc[index, 'AwayTeam_RecentForm'] = sum(g['points'] for g in past_N_games_away) / N


        # Adicionar estatísticas do jogo atual ao histórico da equipe da casa
        # (perspectiva: jogo que acabou de jogar em casa)
        home_points_current_match = get_match_points(row['FTR'], 'home')
        home_team_stats[home_team].append({
            'points': home_points_current_match,
            'goals_scored': row['FTHG'], # Gols que marcou como time da casa
            'goals_conceded': row['FTAG'] # Gols que sofreu como time da casa
        })

        # Adicionar estatísticas do jogo atual ao histórico da equipe visitante
        # (perspectiva: jogo que acabou de jogar fora)
        away_points_current_match = get_match_points(row['FTR'], 'away')
        away_team_stats[away_team].append({
            'points': away_points_current_match,
            'goals_scored': row['FTAG'], # Gols que marcou como time visitante
            'goals_conceded': row['FTHG'] # Gols que sofreu como time visitante
        })

    return df_copy

# --- Fim das Funções Auxiliares ---


# Padrão para encontrar os arquivos CSV baixados no diretório ./app/
FILE_PATTERN = "./app/E0_*.csv"
dataframes = []

print("Starting data processing for Asian Handicap (AH) - 'Lay the Zebra' strategy (with Rolling Features)...")

csv_files = sorted(glob.glob(FILE_PATTERN))

if not csv_files:
    print(f"No CSV files found in ./app/ matching pattern {FILE_PATTERN}. Ensure downloader.py has run successfully.")
    exit() # Modificado para exit()
else:
    print(f"Found the following CSV files to process: {csv_files}")
    for filename in csv_files:
        print(f"Processing file: {filename}")
        try:
            df = pd.read_csv(filename, encoding='ISO-8859-1')
            base_name = os.path.basename(filename)
            season_code_str = base_name.replace('E0_', '').replace('.csv', '')
            season_start_year = pd.NA
            if season_code_str == "2021":
                season_start_year = 2020
            elif len(season_code_str) == 4 and season_code_str.isdigit():
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

# --- Início da Lógica Específica para "Lay à Zebra" ---

# 0. Ordenação e Limpeza Inicial (Necessário ANTES de get_rolling_features)
print("Performing initial sort and data type conversions for rolling features...")
if 'Date' in combined_df.columns:
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce', dayfirst=True)
else:
    print("Critical Error: 'Date' column missing. Cannot proceed with rolling features.")
    exit()

# Season também precisa ser numérica para ordenação correta
if 'Season' in combined_df.columns:
    combined_df['Season'] = pd.to_numeric(combined_df['Season'], errors='coerce') # pd.NA será Coerce
else:
    print("Critical Error: 'Season' column missing. Cannot proceed.")
    exit()

# FTHG, FTAG, FTR são cruciais para get_rolling_features
for col in ['FTHG', 'FTAG']:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    else:
        print(f"Critical Error: Column '{col}' missing. Cannot calculate rolling features.")
        exit()
if 'FTR' not in combined_df.columns:
    print("Critical Error: 'FTR' column missing. Cannot calculate rolling features.")
    exit()

# Drop rows onde Date, Season, FTHG, FTAG, FTR são NaN, pois são essenciais para rolling features
initial_essential_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
combined_df.dropna(subset=initial_essential_cols, inplace=True)
combined_df.sort_values(by=['Season', 'Date'], inplace=True) # Ordenar ANTES de rolling features
print(f"Rows after initial essential NaN drop and sort: {len(combined_df)}")


# A. Calcular Features de Rolling Average (APÓS ordenação e limpeza inicial)
print("Calculating rolling features...")
combined_df = get_rolling_features(combined_df, N=5) # N=5 jogos recentes
print(f"Shape after adding rolling features: {combined_df.shape}")


# 1. Selecionar e Renomear Colunas Essenciais de Odds e AH (Mantido como antes)
odds_cols_map = {
    'B365H': 'B365H', 'B365D': 'B365D', 'B365A': 'B365A',
    'PSCH': 'PSCH', 'PSCA': 'PSCA',
    'PCAHH': 'PCAHH', 'PCAHA': 'PCAHA'
}
for col_original, col_new in odds_cols_map.items():
    if col_original not in combined_df.columns:
        print(f"Warning: Odds column {col_original} not found. Will be NaN.")
        combined_df[col_new] = np.nan
    elif col_original != col_new:
        combined_df[col_new] = combined_df[col_original]

# Colunas a manter: Base + Odds + Rolling Features
cols_to_keep_base = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
cols_to_keep_odds_ah = ['B365H', 'B365D', 'B365A', 'PSCH', 'PSCA', 'PCAHH', 'PCAHA']
cols_to_keep_rolling = [ # Nomes exatos gerados por get_rolling_features
    'HomeTeam_RecentPoints', 'HomeTeam_RecentGoalsScored', 'HomeTeam_RecentGoalsConceded', 'HomeTeam_RecentForm',
    'AwayTeam_RecentPoints', 'AwayTeam_RecentGoalsScored', 'AwayTeam_RecentGoalsConceded', 'AwayTeam_RecentForm'
]
cols_to_keep = cols_to_keep_base + cols_to_keep_odds_ah + cols_to_keep_rolling

final_cols = [col for col in cols_to_keep if col in combined_df.columns]
combined_df = combined_df[final_cols] # Manter apenas as colunas relevantes

# 2. Conversão de Tipos e Tratamento de NaNs para Odds e AH (Mantido como antes)
# As rolling features já são numéricas ou NaN.
numeric_cols_from_source = ['B365H', 'B365D', 'B365A', 'PSCH', 'PSCA', 'PCAHH', 'PCAHA']
for col in numeric_cols_from_source:
    if col in combined_df.columns: # Algumas podem não existir se não estavam nos CSVs
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    else:
        combined_df[col] = np.nan


# 3. Identificar a Zebra (Mantido como antes)
# Odds de Match são críticas para identificar zebra
critical_for_zebra_id = ['B365H', 'B365D', 'B365A', 'FTR'] # FTR é usado para a target
print(f"Rows before dropping NaNs for Zebra ID: {len(combined_df)}")
combined_df.dropna(subset=critical_for_zebra_id, inplace=True)
print(f"Rows after dropping NaNs for Zebra ID: {len(combined_df)}")

combined_df['Zebra'] = np.where(combined_df['B365H'] > combined_df['B365A'], 'H', 'A')
combined_df['Zebra'] = np.where(combined_df['B365H'] == combined_df['B365A'], 'A', combined_df['Zebra'])

# 4. Criar Variável Alvo: ZebraLosesOrDraws (Mantido como antes)
conditions_zebra_loses_or_draws = [
    (combined_df['Zebra'] == 'H') & (combined_df['FTR'].isin(['A', 'D'])),
    (combined_df['Zebra'] == 'A') & (combined_df['FTR'].isin(['H', 'D']))
]
choices = [1, 1]
combined_df['ZebraLosesOrDraws'] = np.select(conditions_zebra_loses_or_draws, choices, default=0)

# 5. Tratamento Final de NaNs (AGORA INCLUI ROLLING FEATURES)
# As colunas de AH (PSCH, PSCA, PCAHH, PCAHA) e as Rolling Features são features, então NaNs nelas devem ser tratados.
# Se optarmos por não usar linhas com NaNs nessas features:
all_feature_cols_to_check_na = cols_to_keep_odds_ah + cols_to_keep_rolling
# Remover apenas colunas que realmente existem no df para evitar KeyErrors no dropna
all_feature_cols_to_check_na_existing = [col for col in all_feature_cols_to_check_na if col in combined_df.columns]

if all_feature_cols_to_check_na_existing:
    print(f"Rows before dropping NaNs from all feature columns (Odds, AH, Rolling): {len(combined_df)}")
    combined_df.dropna(subset=all_feature_cols_to_check_na_existing, inplace=True)
    print(f"Rows after dropping NaNs from all feature columns: {len(combined_df)}")


# 6. Conversões Finais de Tipo e Ordenação (Mantido como antes, mas ordenação já foi feita)
if 'FTHG' in combined_df.columns: combined_df['FTHG'] = combined_df['FTHG'].astype(int)
if 'FTAG' in combined_df.columns: combined_df['FTAG'] = combined_df['FTAG'].astype(int)
# Season e Date já foram tratados e ordenados

# Adicionar a nova variável alvo à lista final de colunas, se ainda não estiver
# E a coluna Zebra
final_selected_cols = cols_to_keep_base + cols_to_keep_odds_ah + cols_to_keep_rolling + ['Zebra', 'ZebraLosesOrDraws']
# Garantir que todas as colunas selecionadas realmente existem
final_selected_cols_existing = [col for col in final_selected_cols if col in combined_df.columns]
combined_df = combined_df[final_selected_cols_existing]


# --- Fim da Lógica Específica para "Lay à Zebra" ---

output_filepath = './app/processed_data_ah.csv' # Mesmo output, mas agora com mais features
combined_df.to_csv(output_filepath, index=False)

print(f"Data processed for AH 'Lay the Zebra' strategy (with Rolling Features) and saved to {output_filepath}")
print("Head of the processed data:")
print(combined_df.head())
if 'ZebraLosesOrDraws' in combined_df.columns:
    print("Value counts for 'ZebraLosesOrDraws':")
    print(combined_df['ZebraLosesOrDraws'].value_counts(normalize=True))
print(f"Shape of processed data: {combined_df.shape}")

print("Data processing script for Asian Handicap (AH) - 'Lay the Zebra' (with Rolling Features) finished.")

```
