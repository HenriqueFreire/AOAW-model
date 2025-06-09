import subprocess
import sys # Para garantir que estamos usando o mesmo interpretador Python

def execute_script(script_name):
    """
    Executa um script Python especificado usando subprocess.
    Verifica se o script foi executado com sucesso.
    """
    print(f"--------------------------------------------------")
    print(f"Iniciando a execução de: {script_name}")
    print(f"--------------------------------------------------")
    try:
        # Usar sys.executable garante que o mesmo interpretador Python que está executando
        # o main.py seja usado para executar os scripts filhos.
        process = subprocess.run(
            [sys.executable, script_name],
            check=True,  # Lança CalledProcessError se o script retornar um código de erro
            capture_output=True, # Captura stdout e stderr
            text=True # Decodifica stdout e stderr como texto
        )
        print(f"Saída de {script_name}:")
        if process.stdout:
            print(process.stdout)
        if process.stderr: # Embora check=True lance erro, pode haver warnings em stderr
            print(f"Saída de erro (stderr) de {script_name}:", file=sys.stderr)
            print(process.stderr, file=sys.stderr)
        print(f"--------------------------------------------------")
        print(f"{script_name} concluído com sucesso.")
        print(f"--------------------------------------------------
")
        return True
    except subprocess.CalledProcessError as e:
        print(f"**************************************************", file=sys.stderr)
        print(f"ERRO ao executar {script_name}!", file=sys.stderr)
        print(f"Código de retorno: {e.returncode}", file=sys.stderr)
        if e.stdout:
            print(f"Saída padrão (stdout) do erro:", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(f"Saída de erro (stderr) do erro:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        print(f"**************************************************
", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"**************************************************", file=sys.stderr)
        print(f"ERRO: O script {script_name} não foi encontrado!", file=sys.stderr)
        print(f"**************************************************
", file=sys.stderr)
        return False

if __name__ == "__main__":
    print("==================================================")
    print("Iniciando o pipeline completo do projeto...")
    print("==================================================
")

    scripts_para_executar = [
        "downloader.py",
        "data_processor.py",
        "model_trainer.py",
        "predictor_lay_aoav.py"
    ]

    for script in scripts_para_executar:
        if not execute_script(script):
            print(f"A execução foi interrompida devido a um erro em {script}.")
            sys.exit(1) # Termina o main.py com um código de erro

    print("==================================================")
    print("Pipeline completo do projeto concluído com sucesso!")
    print("==================================================
")
    sys.exit(0) # Termina o main.py com sucesso
```
