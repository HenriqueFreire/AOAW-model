import pandas as pd
import numpy as np
import os
import glob # Adicionar import do glob

# Padrão para encontrar os arquivos CSV baixados no diretório /app/
# Espera arquivos como E0_2021.csv, E0_2122.csv, E0_2324.csv etc.
FILE_PATTERN = "./app/E0_*.csv"
dataframes = []

print("Starting data processing for LayAOAV target...")

# Lista os arquivos que correspondem ao padrão
csv_files = glob.glob(FILE_PATTERN)
csv_files.sort() # Garante uma ordem consistente, embora a data seja usada para ordenar depois

if not csv_files:
    print(f"No CSV files found in /app/ matching pattern {FILE_PATTERN}. Ensure downloader.py has run successfully.")
else:
    print(f"Found the following CSV files to process: {csv_files}")
    for filename in csv_files:
        # O if os.path.exists(filename) não é estritamente necessário aqui,
        # pois glob.glob só retorna arquivos existentes.
        print(f"Processing file: {filename}")
        try:
            df = pd.read_csv(filename)

            # Extrair código da temporada do nome do arquivo para coluna 'Season'
            base_name = os.path.basename(filename)
            season_code_str = base_name.replace('E0_', '').replace('.csv', '')

            season_start_year = 0
            if season_code_str == "2021": # Temporada 2020/2021
                season_start_year = 2020
            elif len(season_code_str) == 4 and season_code_str.isdigit(): # Ex: 2122, 2223, 2324
                season_start_year = 2000 + int(season_code_str[:2])
            else:
                print(f"Aviso: Não foi possível determinar o ano da temporada para {filename} (código: {season_code_str}).")
                df['Season'] = pd.NA # Usar pd.NA para consistência, será tratado por dropna se Season for crítica

            if season_start_year != 0:
                df['Season'] = season_start_year

            dataframes.append(df)
        except FileNotFoundError: # Should be caught by os.path.exists, but good practice
            print(f"Warning: File {filename} not found during read.")
        except Exception as e:
            print(f"Warning: Could not read {filename} due to error: {e}")
    # else: # This else block is no longer needed as glob only returns existing files
    #     print(f"Warning: File {filename} does not exist. Previous download step might have failed or files are not in /app/.")

if not dataframes:
    print("No dataframes were loaded. Ensure CSV files exist in /app/. Exiting.")
else:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataframes. Total rows before processing: {len(combined_df)}")

    if combined_df.empty:
        print("Combined dataframe is empty. No data to process.")
    else:
        # Define the new target variable LayAOAV
        # Ensure FTAG is numeric and FTR is string before applying conditions.
        print("Converting FTAG to numeric and FTR to string for LayAOAV logic...")
        combined_df['FTAG'] = pd.to_numeric(combined_df['FTAG'], errors='coerce')

        # Drop rows where FTAG could not be converted to numeric or where FTR is NaN, as these are critical for LayAOAV
        combined_df.dropna(subset=['FTAG', 'FTR'], inplace=True)
        print(f"Rows after dropping NaNs from FTAG/FTR: {len(combined_df)}")

        # It's important FTHG is also numeric before any operations, ensure this for other key numeric columns.
        combined_df['FTHG'] = pd.to_numeric(combined_df['FTHG'], errors='coerce')
        # Drop NaNs from FTHG as well, as it's a key result column.
        combined_df.dropna(subset=['FTHG'], inplace=True)
        print(f"Rows after dropping NaNs from FTHG: {len(combined_df)}")


        condition_lose_lay = (combined_df['FTR'].astype(str) == 'A') & (combined_df['FTAG'] >= 4)
        combined_df['LayAOAV'] = np.where(condition_lose_lay, 0, 1)
        print("Defined 'LayAOAV' target variable.")

        # Select relevant columns
        cols_to_keep = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A', 'LayAOAV']

        # Keep only existing columns from cols_to_keep to prevent KeyErrors
        final_cols = [col for col in cols_to_keep if col in combined_df.columns]
        combined_df = combined_df[final_cols]
        print(f"Selected columns. Current columns: {combined_df.columns.tolist()}")

        # Handle missing values for other key textual columns (HomeTeam, AwayTeam)
        # FTR was handled by dropna before LayAOAV. FTHG/FTAG too.
        combined_df.dropna(subset=['HomeTeam', 'AwayTeam'], inplace=True)
        print(f"Rows after dropping NaNs from HomeTeam/AwayTeam: {len(combined_df)}")

        # Garantir que colunas de odds e Season existam antes de tentar tratar NaNs nelas
        cols_for_na_check_odds_season = []
        if 'Season' in combined_df.columns:
            cols_for_na_check_odds_season.append('Season')
        for col in ['B365H', 'B365D', 'B365A']:
            if col in combined_df.columns:
                cols_for_na_check_odds_season.append(col)

        if cols_for_na_check_odds_season:
            print(f"Rows before dropping NaNs from odds/season: {len(combined_df)}")
            combined_df.dropna(subset=cols_for_na_check_odds_season, inplace=True)
            print(f"Rows after dropping NaNs from {cols_for_na_check_odds_season}: {len(combined_df)}")

        # Ensure integer types for goal counts after all processing and NaN drops
        if 'FTHG' in combined_df.columns:
             combined_df['FTHG'] = combined_df['FTHG'].astype(int)
        if 'FTAG' in combined_df.columns:
             combined_df['FTAG'] = combined_df['FTAG'].astype(int)
        print("Ensured integer types for goal columns.")

        # Convert 'Date' column
        if 'Date' in combined_df.columns:
            print("Converting 'Date' column...")
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce', dayfirst=True)
            combined_df.dropna(subset=['Date'], inplace=True) # Drop rows where date conversion failed
            print(f"Rows after 'Date' conversion and NaN drop: {len(combined_df)}")
            # Ensure Season is numeric before sorting if it exists. The dropna for Season should handle pd.NA.
            if 'Season' in combined_df.columns:
                combined_df['Season'] = pd.to_numeric(combined_df['Season'], errors='coerce')
                combined_df.dropna(subset=['Season'], inplace=True) # Drop if conversion failed
                print(f"Rows after 'Season' numeric conversion and NaN drop: {len(combined_df)}")
                combined_df.sort_values(by=['Season', 'Date'], inplace=True)
                print("Sorted DataFrame by 'Season' and 'Date'.")
            else:
                # Fallback to sorting by Date only if Season column is not present
                combined_df.sort_values(by='Date', inplace=True)
                print("Sorted DataFrame by 'Date' (Season column not found).")
        else:
            print("Warning: 'Date' column not found for conversion and sorting.")

        output_filepath = './app/processed_data_lay_aoav.csv'
        combined_df.to_csv(output_filepath, index=False)

        print(f"Data processed with LayAOAV target and saved to {output_filepath}")
        print("Head of the processed data:")
        print(combined_df.head())
        if 'LayAOAV' in combined_df.columns:
            print("Value counts for 'LayAOAV':")
            print(combined_df['LayAOAV'].value_counts(normalize=True))
        print(f"Shape of processed data: {combined_df.shape}")

print("Data processing script for LayAOAV finished.")
