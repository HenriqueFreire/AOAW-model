import pandas as pd
import numpy as np
import joblib
import os

def run_backtesting():
    print("\n--- Iniciando Backtesting para AH 'Lay the Zebra' (Features: Match Odds Only) ---")

    # 1. Carregar Modelo AH e Dados AH
    model_path = './app/model_asian_handicap.joblib' # Modelo treinado com Match Odds Only
    data_path = './app/processed_data_ah.csv'     # Dados processados com Match Odds Only

    if not os.path.exists(model_path):
        print(f"Erro: Modelo AH não encontrado em {model_path}. Execute o treinamento do modelo AH primeiro.")
        return
    if not os.path.exists(data_path):
        print(f"Erro: Dados AH processados não encontrados em {data_path}. Execute o processamento de dados AH primeiro.")
        return

    try:
        model = joblib.load(model_path)
        print(f"Modelo AH carregado de {model_path}")
    except Exception as e:
        print(f"Erro ao carregar o modelo AH: {e}")
        return

    try:
        df_full = pd.read_csv(data_path)
        print(f"Dados AH carregados de {data_path}. Shape: {df_full.shape}")
    except Exception as e:
        print(f"Erro ao carregar os dados AH: {e}")
        return

    # Features que o modelo espera (Apenas Match Odds)
    match_odds_features = ['B365H', 'B365D', 'B365A']

    # Colunas necessárias para o backtest (Season, Match Odds Features, Target)
    required_cols = ['Season'] + match_odds_features + ['ZebraLosesOrDraws']

    missing_cols = [col for col in required_cols if col not in df_full.columns]
    if missing_cols:
        print(f"Erro: Colunas faltando nos dados AH processados: {missing_cols}. Verifique se data_processor.py (Match Odds version) está correto.")
        return

    # Garantir que colunas de features sejam numéricas e tratar NaNs
    for col in match_odds_features + ['Season']:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce')

    cols_to_check_na = ['Season'] + match_odds_features + ['ZebraLosesOrDraws']
    df_full.dropna(subset=cols_to_check_na, inplace=True)

    if df_full.empty:
        print("Dados AH vazios após tratamento de NaNs em features/target/season.")
        return

    df_full['Season'] = df_full['Season'].astype(int)

    # 2. Dividir Dados: Treino (para referência) vs Backtest
    unique_seasons = sorted(df_full['Season'].unique())
    if len(unique_seasons) < 3:
        print(f"Erro: Menos de três temporadas únicas disponíveis nos dados AH ({len(unique_seasons)} encontradas).")
        return

    training_seasons_ref = unique_seasons[:2]
    backtest_seasons = unique_seasons[2:]

    print(f"Temporadas de referência (treino do modelo AH): {training_seasons_ref}")
    print(f"Temporadas para backtesting AH: {backtest_seasons}")

    df_backtest = df_full[df_full['Season'].isin(backtest_seasons)]

    if df_backtest.empty:
        print("Nenhum dado AH disponível para backtesting após seleção das temporadas.")
        return

    print(f"Número de jogos para backtesting AH: {len(df_backtest)}")

    # 3. Executar Lógica de Backtesting
    bets_suggested = 0
    correct_suggestions = 0

    X_backtest = df_backtest[match_odds_features] # Usar apenas Match Odds features
    y_actual_target = df_backtest['ZebraLosesOrDraws'].astype(int)

    if X_backtest.empty:
        print("Não há dados em X_backtest para fazer predições (após filtros e seleção de temporada).")
        return

    try:
        predictions = model.predict(X_backtest)
    except ValueError as ve:
        print(f"Erro ao tentar fazer predições: {ve}")
        print("Isso pode ocorrer se as features em X_backtest não corresponderem às features com as quais o modelo foi treinado.")
        # Tentar imprimir as features esperadas pelo modelo se o scikit-learn permitir (versões mais novas)
        if hasattr(model, 'feature_names_in_'):
            print(f"Features esperadas pelo modelo: {model.feature_names_in_}")
        elif hasattr(model, 'n_features_in_'):
             print(f"Número de features esperadas pelo modelo: {model.n_features_in_}")
        print(f"Features fornecidas para X_backtest: {X_backtest.columns.tolist()}")
        return

    for i in range(len(df_backtest)):
        prediction = predictions[i]
        actual_result = y_actual_target.iloc[i]

        if prediction == 1:
            bets_suggested += 1
            if actual_result == 1:
                correct_suggestions += 1

    # 4. Calcular Métricas
    print("\n--- Resultados do Backtesting AH ('Lay the Zebra' - Features: Match Odds Only) ---")
    if bets_suggested == 0:
        print("Nenhuma sugestão de 'Lay à Zebra' (ZebraLosesOrDraws=1) foi feita pelo modelo durante o período de backtesting.")
        print(f"Total de Jogos no Período de Backtest: {len(df_backtest)}")
        if 'predictions' in locals() and len(predictions) > 0:
             print(f"Contagem de predições do modelo (0:Zebra Ganha, 1:Zebra Não Ganha): {pd.Series(predictions).value_counts().to_dict()}")
        return

    hit_rate = (correct_suggestions / bets_suggested) * 100 if bets_suggested > 0 else 0

    print(f"Total de Jogos no Período de Backtest: {len(df_backtest)}")
    print(f"Número Total de Sugestões 'Lay à Zebra' (modelo previu ZebraLosesOrDraws=1): {bets_suggested}")
    print(f"Número de Sugestões Corretas (Acertos): {correct_suggestions}")
    print(f"Taxa de Acerto: {hit_rate:.2f}%")

if __name__ == '__main__':
    run_backtesting()

```
