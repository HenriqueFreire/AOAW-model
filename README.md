# Configuração e Execução do Projeto com Nix

Este documento fornece instruções sobre como configurar e executar este projeto usando Nix e o arquivo `shell.nix` fornecido.

## Pré-requisitos

1.  **Instalar o Nix:**
    Se você não tem o Nix instalado, siga as instruções no [site oficial do Nix](https://nixos.org/download.html). A instalação multiusuário é geralmente recomendada.
2.  **Conexão com a Internet:**
    Necessária para o download inicial dos dados de futebol durante a execução do `main.py` ou `downloader.py`.

## Entrando no Ambiente de Desenvolvimento

Com o Nix instalado, navegue até o diretório raiz do projeto (onde o arquivo `shell.nix` está localizado) no seu terminal e execute:

```bash
nix-shell
```

Este comando irá construir ou baixar as dependências necessárias definidas no `shell.nix` e o colocará em um novo shell com essas dependências disponíveis. Você deverá ver uma mensagem de boas-vindas do `shellHook`.

## Executando o Pipeline Completo do Projeto

Para executar todo o pipeline do projeto (download de dados, processamento, treinamento do modelo e predição), utilize o script `main.py`:

```bash
python main.py
```

Este script orquestrará a execução dos seguintes componentes em sequência:
1.  `downloader.py`: Baixa os dados históricos de futebol.
2.  `data_processor.py`: Processa os dados baixados.
3.  `model_trainer.py`: Treina o modelo com os dados processados.
4.  `predictor_lay_aoav.py`: Realiza predições usando o modelo treinado.

O `main.py` mostrará o progresso e informará se cada etapa foi concluída com sucesso. Se ocorrer um erro em qualquer etapa, o pipeline será interrompido.

## Execução Manual de Etapas Individuais (Opcional)

Se você precisar executar apenas partes específicas do pipeline (por exemplo, para depuração ou se os dados já foram baixados e processados), você pode executar os scripts individualmente:

### 1. Baixar os Dados Históricos
```bash
python downloader.py
```
Este script buscará os dados das temporadas de futebol da Premier League (E0) desde 2020/2021 até a temporada atual, diretamente do site `football-data.co.uk`, e os salvará no diretório `/app/`.

### 2. Processar os Dados
```bash
python data_processor.py
```
Este script irá procurar os arquivos `E0_*.csv` no diretório `/app/`, processá-los e salvar um arquivo final `processed_data_lay_aoav.csv` também em `/app/`.

### 3. Treinar o Modelo
```bash
python model_trainer.py
```
*(Ajuste o nome do script se for diferente ou se ele requerer argumentos específicos.)*

### 4. Fazer Predições
```bash
python predictor_lay_aoav.py
```
*(Ajuste o nome do script se for diferente ou se ele requerer argumentos específicos.)*

## Notas Adicionais

*   O arquivo `shell.nix` fornece as principais dependências Python (requests, pandas, scikit-learn, joblib).
*   Como mencionado na mensagem do `shellHook`, se você preferir usar o `pip` para gerenciar pacotes adicionais ou para ambientes virtuais mais isolados dentro do Nix shell, você pode criar e ativar um ambiente virtual Python padrão:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt # Ou instale outros pacotes
    ```
    Lembre-se que as dependências principais já são fornecidas pelo Nix, então usar o `pip` para elas pode ser redundante, a menos que você precise de versões muito específicas não gerenciadas pelos seus canais Nix.
```
