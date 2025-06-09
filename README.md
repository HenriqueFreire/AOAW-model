# Configuração e Execução do Projeto com Nix

Este documento fornece instruções sobre como configurar e executar este projeto usando Nix e o arquivo `shell.nix` fornecido.

## Pré-requisitos

1.  **Instalar o Nix:**
    Se você não tem o Nix instalado, siga as instruções no [site oficial do Nix](https://nixos.org/download.html). A instalação multiusuário é geralmente recomendada.
2.  **Conexão com a Internet:**
    Necessária para o download inicial dos dados de futebol (executado através do `main.py` ou diretamente pelo `downloader.py`).

## Entrando no Ambiente de Desenvolvimento

Com o Nix instalado, navegue até o diretório raiz do projeto (onde o arquivo `shell.nix` está localizado) no seu terminal e execute:

```bash
nix-shell
```
Este comando irá construir ou baixar as dependências necessárias e o colocará em um novo shell com o ambiente configurado.

## Usando o Projeto com `main.py`

O script `main.py` é o ponto de entrada principal para interagir com o projeto. Ele oferece um menu interativo para executar as diferentes etapas do pipeline.

Para iniciar, execute:
```bash
python main.py
```

Você verá o seguinte menu:

```
--- Menu Principal ---
1. (Re)Baixar e (Re)Processar Dados
2. (Re)Treinar Modelo para Backtesting (2 primeiras temporadas)
3. Executar Backtesting
4. Consultar Jogo Específico
5. Executar Pipeline Completo (Passos 1, 2, 3)
0. Sair
Escolha uma opção:
```

### Descrição das Opções do Menu:

**Opção 1: (Re)Baixar e (Re)Processar Dados**
*   Executa o `downloader.py` para baixar os dados históricos de partidas da Premier League (desde a temporada 2020/2021) do site `football-data.co.uk`. Os arquivos são salvos em `./app/`.
*   Em seguida, executa o `data_processor.py` que:
    *   Combina todos os CSVs baixados.
    *   Cria uma coluna `Season` (representando o ano de início da temporada, ex: 2020 para a temporada 2020/2021) para cada jogo.
    *   Inclui as odds de Match Odds da Bet365 (`B365H`, `B365D`, `B365A`) como colunas.
    *   Processa os dados e cria a variável alvo `LayAOAV`.
    *   Salva o resultado em `./app/processed_data_lay_aoav.csv`.

**Opção 2: (Re)Treinar Modelo para Backtesting**
*   Executa o `model_trainer.py`.
*   Este script carrega `./app/processed_data_lay_aoav.csv`.
*   Ele treina um modelo de Regressão Logística usando **apenas os dados das duas primeiras temporadas** disponíveis.
*   As features utilizadas para o treinamento são as odds `B365H`, `B365D`, `B365A`.
*   O modelo treinado é salvo em `./app/model_lay_aoav_backtest.joblib`.

**Opção 3: Executar Backtesting**
*   Executa o `backtester.py`.
*   Carrega o modelo de `./app/model_lay_aoav_backtest.joblib` e os dados de `./app/processed_data_lay_aoav.csv`.
*   Usa os dados de todas as temporadas **exceto as duas primeiras** (que foram usadas para treino) para o backtest.
*   **Lógica das Sugestões de Aposta (para avaliação de acerto):**
    *   O modelo prevê `LayAOAV` (1 para sugerir uma aposta Lay, 0 para não sugerir).
    *   Uma "aposta sugerida" é contada sempre que o modelo prevê `LayAOAV = 1`.
    *   Um "acerto" é contado se o modelo previu `LayAOAV = 1` e o resultado real do jogo também foi `LayAOAV = 1`.
*   **Métricas de Saída do Backtest:**
    *   Número Total de Jogos no Período de Backtest.
    *   Número Total de Apostas Sugeridas (LayAOAV=1).
    *   Número de Apostas Corretas (Acertos).
    *   Taxa de Acerto (%).

**Opção 4: Consultar Jogo Específico**
*   Permite que você insira as odds de Match Odds (B365H, B365D, B365A) para um jogo hipotético.
*   Carrega o modelo `./app/model_lay_aoav_backtest.joblib`.
*   Exibe as probabilidades previstas pelo modelo para `LayAOAV=0` e `LayAOAV=1`, e uma sugestão de aposta.

**Opção 5: Executar Pipeline Completo**
*   Executa as opções 1, 2 e 3 em sequência. Ideal para uma primeira execução ou para reprocessar tudo.

**Opção 0: Sair**
*   Encerra o script `main.py`.

## Execução Manual de Scripts Individuais (Para Desenvolvedores/Depuração)

Embora o `main.py` seja a forma recomendada de usar o projeto, os scripts individuais podem ser executados separadamente, se necessário, respeitando suas dependências:
1.  `python downloader.py`: Baixa os dados.
2.  `python data_processor.py`: Processa os dados baixados. (Depende do `downloader.py`)
3.  `python model_trainer.py`: Treina o modelo de backtest. (Depende do `data_processor.py`)
4.  `python backtester.py`: Executa o backtest. (Depende do `model_trainer.py` e `data_processor.py`)

## Notas Adicionais

*   O arquivo `shell.nix` garante que todas as dependências Python (como `pandas`, `scikit-learn`, `joblib`) estejam disponíveis no ambiente Nix.
*   Para ambientes fora do Nix, você pode precisar criar um ambiente virtual Python e instalar as dependências listadas em `requirements.txt` (se este arquivo for mantido).

```
