import subprocess
import sys
import os # Necessário para verificar existência de arquivos e carregar modelo para consulta
import joblib # Necessário para carregar modelo para consulta
import pandas as pd # Necessário para criar DataFrame para consulta
import numpy as np # Se precisarmos de alguma manipulação numérica para consulta

def execute_script(script_name):
    """Executa um script Python e retorna True em sucesso, False em falha."""
    print(f"--- Executando {script_name} ---")
    try:
        process = subprocess.run(
            [sys.executable, script_name],
            check=True, capture_output=True, text=True
        )
        if process.stdout:
            print(process.stdout)
        if process.stderr: # Imprimir stderr mesmo se o script não falhar (para warnings)
             print(f"Saída de erro (stderr) de {script_name}:\n{process.stderr}", file=sys.stderr)
        print(f"--- {script_name} concluído com sucesso ---\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"*** ERRO ao executar {script_name}! Código: {e.returncode} ***", file=sys.stderr)
        if e.stdout:
            print(f"Stdout do erro:\n{e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"Stderr do erro:\n{e.stderr}", file=sys.stderr)
        print(f"--- Falha em {script_name} ---\n", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"*** ERRO: Script {script_name} não encontrado! ***", file=sys.stderr)
        print(f"--- Falha em {script_name} ---\n", file=sys.stderr)
        return False

def consult_specific_game():
    print("\n--- Consulta de Jogo Específico ---")
    model_path = './app/model_lay_aoav_backtest.joblib' # Usando o modelo de backtest por enquanto

    if not os.path.exists(model_path):
        print(f"Modelo não encontrado em {model_path}. Treine o modelo primeiro (Opção 2 do menu).")
        return

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    print("Por favor, forneça as odds de Match Odds (formato decimal, ex: 2.50):")

    try:
        b365h = float(input("Odd B365H (Vitória Casa): "))
        b365d = float(input("Odd B365D (Empate): "))
        b365a = float(input("Odd B365A (Vitória Fora): "))
    except ValueError:
        print("Entrada inválida. Por favor, insira números decimais para as odds.")
        return

    # As features devem corresponder exatamente às usadas no treinamento do modelo carregado
    # model_trainer.py para model_lay_aoav_backtest.joblib usa ['B365H', 'B365D', 'B365A']
    feature_names = ['B365H', 'B365D', 'B365A']
    game_features = pd.DataFrame([[b365h, b365d, b365a]], columns=feature_names)

    try:
        prediction_proba = model.predict_proba(game_features)[0]
        prediction = model.predict(game_features)[0]

        print(f"\n--- Predição do Modelo ({model_path}) ---")
        print(f"Features fornecidas: B365H={b365h}, B365D={b365d}, B365A={b365a}")
        print(f"Probabilidade de LayAOAV=0 (Condição de Perda do Lay Ocorre): {prediction_proba[0]:.4f}")
        print(f"Probabilidade de LayAOAV=1 (Condição de Perda do Lay NÃO Ocorre): {prediction_proba[1]:.4f}")

        if prediction == 1:
            print("Decisão Sugerida: Lay AOAV (Apostar que a Vitória Fora com >=4 gols NÃO acontecerá).")
        else:
            print("Decisão Sugerida: Não Apostar Lay AOAV (Risco da Vitória Fora com >=4 gols é considerado alto).")
        print("--------------------------------------")

    except Exception as e:
        print(f"Erro durante a predição: {e}")


def main_menu():
    while True:
        print("\n--- Menu Principal ---")
        print("1. (Re)Baixar e (Re)Processar Dados")
        print("2. (Re)Treinar Modelo para Backtesting (2 primeiras temporadas)")
        print("3. Executar Backtesting")
        print("4. Consultar Jogo Específico")
        print("5. Executar Pipeline Completo (Passos 1, 2, 3)")
        print("0. Sair")

        choice = input("Escolha uma opção: ")

        if choice == '1':
            if execute_script('downloader.py'):
                execute_script('data_processor.py')
        elif choice == '2':
            execute_script('model_trainer.py')
        elif choice == '3':
            execute_script('backtester.py')
        elif choice == '4':
            consult_specific_game()
        elif choice == '5':
            print("--- Iniciando Pipeline Completo ---")
            if execute_script('downloader.py'):
                if execute_script('data_processor.py'):
                    if execute_script('model_trainer.py'):
                        execute_script('backtester.py')
            print("--- Pipeline Completo Finalizado (verifique erros acima se houver) ---")
        elif choice == '0':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main_menu()
```
