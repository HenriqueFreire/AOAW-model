import subprocess
import sys
import os
import joblib
import pandas as pd
# numpy não é explicitamente usado aqui, mas pandas pode usá-lo internamente.

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
        if process.stderr:
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

def consult_specific_game_ah():
    print("\n--- Consulta de Jogo Específico (Modelo 'Lay à Zebra') ---")
    model_path = './app/model_asian_handicap.joblib'

    if not os.path.exists(model_path):
        print(f"Modelo AH não encontrado em {model_path}. Treine o modelo primeiro (Opção 2 do menu).")
        return

    try:
        model = joblib.load(model_path)
        print(f"Modelo AH carregado: {model_path}")
    except Exception as e:
        print(f"Erro ao carregar o modelo AH: {e}")
        return

    print("Por favor, forneça as features para o jogo (formato decimal, ex: 2.50):")

    feature_names_ah = ['B365H', 'B365D', 'B365A', 'PSCH', 'PSCA', 'PCAHH', 'PCAHA']
    game_feature_values = []

    try:
        for feature_name in feature_names_ah:
            value = float(input(f"Feature '{feature_name}': "))
            game_feature_values.append(value)
    except ValueError:
        print("Entrada inválida. Por favor, insira números decimais para as features.")
        return

    game_features_df = pd.DataFrame([game_feature_values], columns=feature_names_ah)

    try:
        prediction_proba = model.predict_proba(game_features_df)[0]
        prediction = model.predict(game_features_df)[0]

        print(f"\n--- Predição do Modelo AH ({model_path}) ---")
        print(f"Features fornecidas: {game_features_df.to_dict(orient='records')[0]}")
        print(f"Probabilidade de ZebraLosesOrDraws=0 (Zebra Vence): {prediction_proba[0]:.4f}")
        print(f"Probabilidade de ZebraLosesOrDraws=1 (Zebra Perde ou Empata): {prediction_proba[1]:.4f}")

        if prediction == 1:
            print("Decisão Sugerida (Lay à Zebra): A Zebra NÃO VENCERÁ (perderá ou empatará).")
        else:
            print("Decisão Sugerida (Lay à Zebra): A Zebra PODERÁ VENCER.")
        print("--------------------------------------")

    except Exception as e:
        print(f"Erro durante a predição AH: {e}")


def main_menu():
    while True:
        print("\n--- Menu Principal (Foco: Modelo Handicap Asiático 'Lay à Zebra') ---")
        print("1. (Re)Baixar Dados e (Re)Processar para AH")
        print("2. (Re)Treinar Modelo AH (2 primeiras temporadas)")
        print("3. Executar Backtesting do Modelo AH (Taxa de Acerto)")
        print("4. Consultar Jogo Específico (Modelo AH)")
        print("5. Executar Pipeline AH Completo (Passos 1, 2, 3)")
        print("0. Sair")

        choice = input("Escolha uma opção: ")

        if choice == '1':
            if execute_script('downloader.py'): # downloader.py permanece o mesmo
                execute_script('data_processor.py') # data_processor.py agora é focado em AH
        elif choice == '2':
            execute_script('model_trainer.py') # model_trainer.py agora é focado em AH
        elif choice == '3':
            execute_script('backtester.py')   # backtester.py agora é focado em AH
        elif choice == '4':
            consult_specific_game_ah() # Nova função de consulta para AH
        elif choice == '5':
            print("--- Iniciando Pipeline AH Completo ---")
            if execute_script('downloader.py'):
                if execute_script('data_processor.py'):
                    if execute_script('model_trainer.py'):
                        execute_script('backtester.py')
            print("--- Pipeline AH Completo Finalizado (verifique erros acima se houver) ---")
        elif choice == '0':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main_menu()
```
