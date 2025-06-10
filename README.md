# Projeto de Modelo Preditivo para Handicap Asiático ("Lay à Zebra") com Nix

Este documento fornece instruções sobre como configurar e executar este projeto, que visa construir um modelo preditivo para a estratégia "Lay à Zebra" em jogos de futebol, utilizando dados da Premier League.

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

O script `main.py` é o ponto de entrada principal para interagir com o projeto. Ele oferece um menu interativo para executar as diferentes etapas do pipeline focado na estratégia "Lay à Zebra".

Para iniciar, execute:
```bash
python main.py
```

Você verá o seguinte menu (ou similar, refletindo as opções AH):
```
--- Menu Principal (Foco: Modelo Handicap Asiático 'Lay à Zebra') ---
1. (Re)Baixar Dados e (Re)Processar para AH
2. (Re)Treinar Modelo AH (2 primeiras temporadas)
3. Executar Backtesting do Modelo AH (Taxa de Acerto)
4. Consultar Jogo Específico (Modelo AH)
5. Executar Pipeline AH Completo (Passos 1, 2, 3)
0. Sair
Escolha uma opção:
```

### Descrição das Opções do Menu:

**Opção 1: (Re)Baixar Dados e (Re)Processar para AH**
*   Executa o `downloader.py` para baixar os dados históricos de partidas da Premier League (desde a temporada 2020/2021) do site `football-data.co.uk`. Os arquivos são salvos em `./app/`.
*   Em seguida, executa o `data_processor.py` que:
    *   Combina todos os CSVs baixados.
    *   Cria uma coluna `Season` (ano de início da temporada).
    *   Mantém e processa colunas de odds de Match Odds (ex: `B365H,D,A`), Linhas de Handicap Asiático (ex: `PSCH, PSCA`), e Odds de Handicap Asiático (ex: `PCAHH, PCAHA`).
    *   Identifica a `Zebra` em cada jogo (time com maior odd de vitória).
    *   Cria a variável alvo `ZebraLosesOrDraws` (1 se a zebra perde ou empata, 0 se a zebra vence).
    *   Salva o resultado em `./app/processed_data_ah.csv`.

**Opção 2: (Re)Treinar Modelo AH**
*   Executa o `model_trainer.py`.
*   Carrega `./app/processed_data_ah.csv`.
*   Treina um modelo de Regressão Logística usando **apenas os dados das duas primeiras temporadas** disponíveis.
*   As **features** utilizadas são: Odds de Match Odds (`B365H,D,A`), Linhas de AH (`PSCH, PSCA`), e Odds de AH (`PCAHH, PCAHA`).
*   O modelo treinado é salvo em `./app/model_asian_handicap.joblib`.

**Opção 3: Executar Backtesting do Modelo AH**
*   Executa o `backtester.py`.
*   Carrega o modelo de `./app/model_asian_handicap.joblib` e os dados de `./app/processed_data_ah.csv`.
*   Usa os dados de todas as temporadas **exceto as duas primeiras** (que foram usadas para treino) para o backtest.
*   **Lógica da Avaliação (Taxa de Acerto):**
    *   O modelo prevê `ZebraLosesOrDraws` (1 para sugerir que a zebra não vencerá, 0 caso contrário).
    *   Uma "sugestão" é contada sempre que o modelo prevê `ZebraLosesOrDraws = 1`.
    *   Um "acerto" é contado se o modelo previu `ZebraLosesOrDraws = 1` e a zebra realmente não venceu.
*   **Métricas de Saída do Backtest:**
    *   Número Total de Jogos no Período de Backtest.
    *   Número Total de Sugestões "Lay à Zebra" (modelo previu `ZebraLosesOrDraws=1`).
    *   Número de Sugestões Corretas (Acertos).
    *   Taxa de Acerto (%).

**Opção 4: Consultar Jogo Específico (Modelo AH)**
*   Permite que você insira as features necessárias para o modelo AH: Odds de Match Odds (`B365H,D,A`), Linhas de AH (`PSCH, PSCA`), e Odds de AH (`PCAHH, PCAHA`).
*   Carrega o modelo `./app/model_asian_handicap.joblib`.
*   Exibe as probabilidades previstas pelo modelo para `ZebraLosesOrDraws` e uma sugestão.

**Opção 5: Executar Pipeline AH Completo**
*   Executa as opções 1, 2 e 3 em sequência.

**Opção 0: Sair**
*   Encerra o script `main.py`.

## Execução Manual de Scripts Individuais (Para Desenvolvedores/Depuração)

Embora o `main.py` seja a forma recomendada, os scripts podem ser executados individualmente, respeitando suas dependências (todos agora focados na estratégia AH):
1.  `python downloader.py`
2.  `python data_processor.py` (Depende do `downloader.py`)
3.  `python model_trainer.py` (Depende do `data_processor.py`)
4.  `python backtester.py` (Depende do `model_trainer.py` e `data_processor.py`)

## Notas Adicionais

*   O arquivo `shell.nix` garante as dependências Python.
*   Considere o uso de um ambiente virtual Python e `requirements.txt` para portabilidade fora do Nix.
```
