import pandas as pd
import numpy as np # Mantido caso alguma manipulação numérica seja necessária no futuro
import joblib
import os

# A função calculate_drawdown não é mais necessária.

def run_backtesting():
    print("\n--- Iniciando Backtesting (Foco: Taxa de Acerto) ---")

    # 1. Carregar Modelo e Dados
    model_path = './app/model_lay_aoav_backtest.joblib'
    data_path = './app/processed_data_lay_aoav.csv'

    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}. Execute o treinamento do modelo para backtest primeiro.")
        return
    if not os.path.exists(data_path):
        print(f"Erro: Dados processados não encontrados em {data_path}. Execute o processamento de dados primeiro.")
        return

    try:
        model = joblib.load(model_path)
        print(f"Modelo carregado de {model_path}")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    try:
        df_full = pd.read_csv(data_path)
        print(f"Dados carregados de {data_path}. Shape: {df_full.shape}")
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return

    required_cols = ['Season', 'B365H', 'B365D', 'B365A', 'LayAOAV']
    missing_cols = [col for col in required_cols if col not in df_full.columns]
    if missing_cols:
        print(f"Erro: Colunas faltando nos dados processados: {missing_cols}")
        return

    for col in ['B365H', 'B365D', 'B365A', 'Season']:
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
    df_full.dropna(subset=['B365H', 'B365D', 'B365A', 'Season', 'LayAOAV'], inplace=True)

    if df_full.empty:
        print("Dados vazios após tratamento de NaNs.")
        return

    df_full['Season'] = df_full['Season'].astype(int)

    # 2. Dividir Dados: Treino (para referência) vs Backtest
    unique_seasons = sorted(df_full['Season'].unique())
    if len(unique_seasons) < 3:
        print(f"Erro: Menos de três temporadas únicas disponíveis ({len(unique_seasons)} encontradas). Não é possível separar dados para backtest.")
        return

    training_seasons_ref = unique_seasons[:2]
    backtest_seasons = unique_seasons[2:]

    print(f"Temporadas de referência (treino do modelo): {training_seasons_ref}")
    print(f"Temporadas para backtesting: {backtest_seasons}")

    df_backtest = df_full[df_full['Season'].isin(backtest_seasons)]

    if df_backtest.empty:
        print("Nenhum dado disponível para backtesting após seleção das temporadas.")
        return

    print(f"Número de jogos para backtesting: {len(df_backtest)}")

    # 3. Executar Lógica de Backtesting (Foco na Taxa de Acerto)
    bets_made = 0
    bets_won = 0

    feature_cols = ['B365H', 'B365D', 'B365A']
    X_backtest = df_backtest[feature_cols]
    y_actual_layaoav = df_backtest['LayAOAV'].astype(int)

    predictions = model.predict(X_backtest)

    for i in range(len(df_backtest)):
        prediction = predictions[i]
        actual_result = y_actual_layaoav.iloc[i]

        if prediction == 1: # Modelo sugere apostar Lay AOAV
            bets_made += 1
            if actual_result == 1: # Aposta Lay AOAV teria sido ganha
                bets_won += 1
            # Não há cálculo de P&L

    # 4. Calcular Métricas (Foco na Taxa de Acerto)
    print("\n--- Resultados do Backtesting (Taxa de Acerto) ---")
    if bets_made == 0:
        print("Nenhuma aposta (LayAOAV=1) foi sugerida pelo modelo durante o período de backtesting.")
        # Imprimir informações adicionais se nenhuma aposta foi feita
        print(f"Total de Jogos no Período de Backtest: {len(df_backtest)}")
        # Poderia-se adicionar a contagem de predições LayAOAV=0 vs LayAOAV=1 aqui se desejado
        # print(f"Contagem de predições do modelo (0:Não Apostar Lay, 1:Apostar Lay): {pd.Series(predictions).value_counts().to_dict()}")
        return

    hit_rate = (bets_won / bets_made) * 100 if bets_made > 0 else 0

    print(f"Total de Jogos no Período de Backtest: {len(df_backtest)}")
    print(f"Número Total de Apostas Sugeridas (LayAOAV=1): {bets_made}")
    print(f"Número de Apostas Corretas (Acertos): {bets_won}")
    print(f"Taxa de Acerto: {hit_rate:.2f}%")

if __name__ == '__main__':
    run_backtesting()

```
