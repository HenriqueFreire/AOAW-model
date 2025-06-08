import pandas as pd
import numpy as np
import os

FILENAMES = ['/app/E0_2324.csv', '/app/E0_2223.csv', '/app/E0_2122.csv']
dataframes = []

print("Starting data processing for LayAOAV target...")

for filename in FILENAMES:
    if os.path.exists(filename):
        print(f"Processing file: {filename}")
        try:
            df = pd.read_csv(filename)
            dataframes.append(df)
        except FileNotFoundError: # Should be caught by os.path.exists, but good practice
            print(f"Warning: File {filename} not found during read.")
        except Exception as e:
            print(f"Warning: Could not read {filename} due to error: {e}")
    else:
        print(f"Warning: File {filename} does not exist. Previous download step might have failed or files are not in /app/.")

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
        cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'LayAOAV']

        # Keep only existing columns from cols_to_keep to prevent KeyErrors
        final_cols = [col for col in cols_to_keep if col in combined_df.columns]
        combined_df = combined_df[final_cols]
        print(f"Selected columns. Current columns: {combined_df.columns.tolist()}")

        # Handle missing values for other key textual columns (HomeTeam, AwayTeam)
        # FTR was handled by dropna before LayAOAV. FTHG/FTAG too.
        combined_df.dropna(subset=['HomeTeam', 'AwayTeam'], inplace=True)
        print(f"Rows after dropping NaNs from HomeTeam/AwayTeam: {len(combined_df)}")

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
            combined_df.sort_values(by='Date', inplace=True)
            print("Sorted DataFrame by 'Date'.")
        else:
            print("Warning: 'Date' column not found for conversion and sorting.")

        output_filepath = '/app/processed_data_lay_aoav.csv'
        combined_df.to_csv(output_filepath, index=False)

        print(f"Data processed with LayAOAV target and saved to {output_filepath}")
        print("Head of the processed data:")
        print(combined_df.head())
        if 'LayAOAV' in combined_df.columns:
            print("Value counts for 'LayAOAV':")
            print(combined_df['LayAOAV'].value_counts(normalize=True))
        print(f"Shape of processed data: {combined_df.shape}")

print("Data processing script for LayAOAV finished.")
