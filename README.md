# Configuração e Execução do Projeto com Nix

Este documento fornece instruções sobre como configurar e executar este projeto usando Nix e o arquivo `shell.nix` fornecido.

## Pré-requisitos

1.  **Instalar o Nix:**
    Se você não tem o Nix instalado, siga as instruções no [site oficial do Nix](https://nixos.org/download.html). A instalação multiusuário é geralmente recomendada.
2.  **Conexão com a Internet:**
    Necessária para o download inicial dos dados de futebol durante a execução do `main.py` via `downloader.py`.

## Entrando no Ambiente de Desenvolvimento

Com o Nix instalado, navegue até o diretório raiz do projeto (onde o arquivo `shell.nix` está localizado) no seu terminal e execute:

```bash
nix-shell
```

Este comando irá construir ou baixar as dependências necessárias definidas no `shell.nix` e o colocará em um novo shell com essas dependências disponíveis. Você deverá ver uma mensagem de boas-vindas do `shellHook`.

## Executando o Pipeline Completo do Projeto

Para executar todo o pipeline do projeto (download de dados, processamento, clustering, treinamento do modelo, predição e backtesting), utilize o script `main.py`:

```bash
python main.py
```

Este script orquestrará a execução dos seguintes componentes em sequência:
1.  `downloader.py`: Baixa os dados históricos de futebol.
2.  `data_processor.py`: Processa os dados baixados, incluindo a criação da variável alvo `LayAOAV` e o processamento das odds de apostas. O resultado é salvo em `/app/processed_data_lay_aoav.csv`.
3.  `cluster_features.py`: Carrega os dados de `processed_data_lay_aoav.csv`, gera `rolling features` (métricas de forma recente das equipas) e aplica o algoritmo k-means para agrupar os jogos com base nessas features. O resultado, incluindo uma coluna `match_cluster_id`, é salvo em `/app/processed_data_clustered.csv`.
4.  `model_trainer.py`: Treina o modelo de classificação para `LayAOAV` utilizando os dados de `/app/processed_data_clustered.csv`. A feature `match_cluster_id` é one-hot encoded e incluída no treino. O modelo treinado é salvo em `/app/model_lay_aoav.joblib`.
5.  `predictor_lay_aoav.py`: Carrega o modelo treinado e os dados de `/app/processed_data_clustered.csv` para simular predições em "novos" jogos, utilizando as `rolling features` e a `match_cluster_id` (one-hot encoded) de forma consistente com o treino.
6.  `backtester.py`: Avalia a performance da estratégia de apostas `LayAOAV` utilizando o modelo treinado e os dados de `/app/processed_data_clustered.csv`, também incorporando a `match_cluster_id` (one-hot encoded) na preparação das features para o backtesting.

O `main.py` mostrará o progresso e informará se cada etapa foi concluída com sucesso. Se ocorrer um erro em qualquer etapa, o pipeline será interrompido.

## Execução Manual de Etapas Individuais (Opcional)

Se você precisar executar apenas partes específicas do pipeline, você pode executar os scripts individualmente:

### 1. Baixar os Dados Históricos
```bash
python downloader.py
```
Este script buscará os dados das temporadas de futebol da Premier League (E0) especificadas em seu código (ex: desde 2020/2021 até a temporada atual), diretamente do site `football-data.co.uk`, e os salvará no diretório `/app/`.

### 2. Processar os Dados
```bash
python data_processor.py
```
Este script irá procurar os arquivos `E0_*.csv` no diretório `/app/`, combiná-los, limpar os dados, criar a variável alvo `LayAOAV` (onde 1 significa que o time visitante NÃO venceu por 4 ou mais gols de diferença), processar colunas de odds de apostas (como `PSCH`, `PSCD`, `PSCA`, `B365CH`, `B365CD`, `B365CA`) convertendo-as para numérico, e salvar um arquivo final `processed_data_lay_aoav.csv` também em `/app/`.

### 3. Gerar Features e Agrupar Jogos (Clustering)
```bash
python cluster_features.py
```
Este script é executado após `data_processor.py`.
**Finalidade:** Carrega `/app/processed_data_lay_aoav.csv`, gera `rolling features` (pontos recentes, golos marcados/sofridos para equipa da casa e visitante nos últimos N jogos) e aplica o algoritmo k-means (com k=4 por defeito) a estas `rolling features` (após normalização com `StandardScaler`) para agrupar os jogos.
**Output:** Salva um novo arquivo `/app/processed_data_clustered.csv`. Este arquivo contém todas as colunas de `processed_data_lay_aoav.csv`, mais as `rolling features` geradas e uma nova coluna `match_cluster_id` que indica o cluster ao qual cada jogo pertence.

### 4. Treinar o Modelo
```bash
python model_trainer.py
```
Este script agora carrega dados de `/app/processed_data_clustered.csv`. As `rolling features` já estão presentes neste arquivo, portanto, as funções para calculá-las foram removidas de `model_trainer.py`. A coluna `match_cluster_id` é tratada como uma feature categórica e é convertida para múltiplas features binárias (one-hot encoding) antes de treinar o modelo de Regressão Logística. O modelo treinado é salvo como `/app/model_lay_aoav.joblib`.

**Nota sobre o Impacto das Novas Features:** A inclusão das `rolling features` (agora geradas explicitamente em `cluster_features.py`) e da feature `match_cluster_id` (one-hot encoded) visa melhorar o poder preditivo do modelo. É importante observar as métricas de avaliação (Accuracy, Classification Report, ROC AUC) impressas por `model_trainer.py` e os resultados do `backtester.py` para determinar o impacto real destas novas features na performance da estratégia `LayAOAV`.

### 5. Fazer Predições (Exemplo)
```bash
python predictor_lay_aoav.py
```
Este script carrega o modelo salvo (`model_lay_aoav.joblib`) e os dados processados de `/app/processed_data_clustered.csv`. Ele simula o processo de predição para os últimos jogos do dataset, utilizando as `rolling features` já presentes e tratando a `match_cluster_id` (via one-hot encoding) de forma consistente com o processo de treino. As funções para gerar `rolling features` foram removidas deste script.

### 6. Avaliar a Estratégia com Backtesting
```bash
python backtester.py
```
Este script é crucial para avaliar a viabilidade da estratégia de apostas `LayAOAV`.
**Propósito:** Testar o desempenho do modelo treinado em dados históricos que ele não viu durante o treinamento.
**Metodologia:**
- Carrega o modelo salvo (`model_lay_aoav.joblib`) e os dados de `/app/processed_data_clustered.csv`. As `rolling features` e `match_cluster_id` já estão presentes.
- Seleciona uma porção dos dados para backtesting (ex: os últimos 20% dos jogos).
- Prepara as features para o backtesting, incluindo o one-hot encoding da `match_cluster_id` de forma consistente com o treino.
- Para cada jogo no conjunto de backtesting, ele obtém a probabilidade prevista pelo modelo para `LayAOAV=1`.
- Simula apostas "Lay AOAV" se esta probabilidade `P(LayAOAV=1)` excede diferentes limiares.
- Calcula e reporta a 'taxa de acerto' (hit rate) para cada limiar.
**Interpretação do Output:**
O script imprimirá uma tabela mostrando, para cada limiar de probabilidade: o número de apostas simuladas, o número de apostas bem-sucedidas e a taxa de acerto. Uma taxa de acerto mais alta com um limiar de probabilidade mais alto pode indicar um sinal mais confiável, mas provavelmente resultará em menos oportunidades de aposta.

## Desenvolvendo Novas Estratégias de Apostas

Esta seção descreve caminhos para desenvolvimento futuro e criação de novas estratégias de apostas baseadas neste projeto.

### 1. Novas Variáveis Alvo (Target Variables)
- **Como:** Modifique `data_processor.py` para definir novas colunas alvo.
- **Impacto:** Exigirá o treinamento de novos modelos em `model_trainer.py`.

### 2. Engenharia Avançada de Features
- **Como:** Aprimore a geração de features em `cluster_features.py` ou adicione novas features em `data_processor.py`.
- **Ideias:** Métricas de forma sobre diferentes janelas, estatísticas H2H, uso de odds como features (com cautela para evitar data leakage), ratings de força da equipe.

### 3. Diferentes Arquiteturas de Modelo
- **Como:** Experimente com outros algoritmos de classificação no `model_trainer.py`.
- **Exemplos:** `RandomForestClassifier`, `GradientBoostingClassifier`, Redes Neurais.

### 4. Lógica de Estratégia Baseada em Odds (Value Betting)
- **Conceito:** Comparar probabilidades do modelo com as implícitas nas odds das casas de apostas.
- **Como:** Utilizar as colunas de odds já presentes em `processed_data_clustered.csv` no `backtester.py`.

### 5. Simulação de Lucro e Perda (P&L)
- **Contexto:** O `backtester.py` atual foca na taxa de acerto. Simular P&L é essencial.
- **Como:** Incorporar gestão de banca e cálculo de retornos baseado nas odds.

### 6. Validação Robusta
- **Importância:** Testar rigorosamente qualquer nova estratégia.
- **Métodos:** Teste Out-of-Sample, Walk-Forward Testing, Paper Trading.

## Notas Adicionais

*   O arquivo `shell.nix` fornece as principais dependências Python.
*   Para ambientes virtuais Python dentro do Nix shell:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt # Ou instale outros pacotes
    ```
