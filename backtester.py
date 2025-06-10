import pandas as pd
import numpy as np
import joblib
import os

def run_backtesting():
    print("\n--- Iniciando Backtesting para AH 'Lay the Zebra' (Foco: Taxa de Acerto) ---")

    # 1. Carregar Modelo AH e Dados AH
    model_path = './app/model_asian_handicap.joblib' # Alterado para o modelo AH
    data_path = './app/processed_data_ah.csv'     # Alterado para os dados AH

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

    # Colunas necessárias para o backtest do modelo AH
    required_cols = [
        'Season',
        'B365H', 'B365D', 'B365A', # Match Odds features
        'PSCH', 'PSCA',             # AH Lines features
        'PCAHH', 'PCAHA',            # AH Odds features
        'ZebraLosesOrDraws'         # Target
    ]
    missing_cols = [col for col in required_cols if col not in df_full.columns]
    if missing_cols:
        print(f"Erro: Colunas faltando nos dados AH processados: {missing_cols}")
        return

    # Features usadas no model_trainer_ah.py
    feature_cols_ah = ['B365H', 'B365D', 'B365A', 'PSCH', 'PSCA', 'PCAHH', 'PCAHA']

    # Garantir que colunas de features sejam numéricas e tratar NaNs
    for col in feature_cols_ah + ['Season']:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
    df_full.dropna(subset=feature_cols_ah + ['Season', 'ZebraLosesOrDraws'], inplace=True)

    if df_full.empty:
        print("Dados AH vazios após tratamento de NaNs.")
        return

    df_full['Season'] = df_full['Season'].astype(int)

    # 2. Dividir Dados: Treino (para referência) vs Backtest
    unique_seasons = sorted(df_full['Season'].unique())
    if len(unique_seasons) < 3: # Precisa de pelo menos 2 para treino e 1 para backtest
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

    # 3. Executar Lógica de Backtesting (Foco na Taxa de Acerto para 'ZebraLosesOrDraws')
    bets_suggested = 0 # Renomeado de bets_made para clareza
    correct_suggestions = 0 # Renomeado de bets_won para clareza

    X_backtest = df_backtest[feature_cols_ah]
    y_actual_target = df_backtest['ZebraLosesOrDraws'].astype(int)

    if X_backtest.empty:
        print("Não há dados em X_backtest para fazer predições.")
        return

    predictions = model.predict(X_backtest) # Modelo prevê ZebraLosesOrDraws = 1 ou 0

    for i in range(len(df_backtest)):
        prediction = predictions[i]
        actual_result = y_actual_target.iloc[i]

        if prediction == 1: # Modelo sugere que a Zebra NÃO VAI GANHAR (ZebraLosesOrDraws=1)
            bets_suggested += 1
            if actual_result == 1: # Zebra realmente não ganhou
                correct_suggestions += 1

    # 4. Calcular Métricas (Foco na Taxa de Acerto para 'ZebraLosesOrDraws')
    print("\n--- Resultados do Backtesting AH ('Lay the Zebra') ---")
    if bets_suggested == 0:
        print("Nenhuma sugestão de 'Lay à Zebra' (ZebraLosesOrDraws=1) foi feita pelo modelo durante o período de backtesting.")
        print(f"Total de Jogos no Período de Backtest: {len(df_backtest)}")
        # Opcional: Contagem de predições 0 vs 1
        # print(f"Contagem de predições do modelo (0:Zebra Ganha, 1:Zebra Não Ganha): {pd.Series(predictions).value_counts().to_dict()}")
        return

    hit_rate = (correct_suggestions / bets_suggested) * 100 if bets_suggested > 0 else 0

    print(f"Total de Jogos no Período de Backtest: {len(df_backtest)}")
    print(f"Número Total de Sugestões 'Lay à Zebra' (modelo previu ZebraLosesOrDraws=1): {bets_suggested}")
    print(f"Número de Sugestões Corretas (Acertos): {correct_suggestions}")
    print(f"Taxa de Acerto: {hit_rate:.2f}%")

if __name__ == '__main__':
    run_backtesting()

```
