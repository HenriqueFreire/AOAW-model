{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.requests
    pkgs.python3Packages.pandas
    pkgs.python3Packages.scikitlearn
    pkgs.python3Packages.joblib
  ];
  shellHook = ''
    echo "Entrando no ambiente de desenvolvimento Python para o projeto de Handicap Asiático."
    echo "Você pode precisar criar e ativar um ambiente virtual separado para instalar pacotes com pip se preferir."
    # Exemplo: python -m venv .venv
    # Exemplo: source .venv/bin/activate
    # Exemplo: pip install -r requirements.txt
    echo "As dependências Python principais (requests, pandas, numpy, scipy) já estão disponíveis via Nix."
  '';
}
