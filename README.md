# Configuração e Execução do Projeto com Nix

Este documento fornece instruções sobre como configurar e executar este projeto usando Nix e o arquivo `shell.nix` fornecido.

## Pré-requisitos

1.  **Instalar o Nix:**
    Se você não tem o Nix instalado, siga as instruções no [site oficial do Nix](https://nixos.org/download.html). A instalação multiusuário é geralmente recomendada.

## Entrando no Ambiente de Desenvolvimento

Com o Nix instalado, navegue até o diretório raiz do projeto (onde o arquivo `shell.nix` está localizado) no seu terminal e execute:

```bash
nix-shell
```

Este comando irá construir ou baixar as dependências necessárias definidas no `shell.nix` e o colocará em um novo shell com essas dependências disponíveis. Você deverá ver uma mensagem de boas-vindas do `shellHook`.

## Executando os Scripts do Projeto

Dentro do Nix shell, você pode executar os scripts Python do projeto como faria normalmente. Aqui estão alguns exemplos:

*   **Para executar o processador de dados:**
    ```bash
    python data_processor.py
    ```

*   **Para executar o treinador de modelo:**
    ```bash
    python model_trainer.py
    ```

*   **Para executar o preditor:**
    ```bash
    python predictor_lay_aoav.py
    ```

    *(Ajuste os nomes dos scripts se forem diferentes ou se exigirem argumentos específicos.)*

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
