import requests
import datetime
import os

# URL base para os dados da Premier League (E0) no football-data.co.uk
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
OUTPUT_DIR = "/app/"

def get_season_codes(start_year=2020):
    """
    Gera códigos de temporada no formato 'YY(YY+1)' (ex: '2021' para 2020-2021)
    e 'YYYY' (ex: '2324' para 2023-2024) a partir do start_year até o ano atual.
    O football-data.co.uk usa formatos diferentes dependendo da temporada.
    Ex: 1920/E0.csv (para 2019-2020), 2021/E0.csv (para 2020-2021)
    Para temporadas mais recentes, parece ser YY(YY+1) ex: 2324/E0.csv
    Vamos precisar verificar o padrão exato para as temporadas que queremos.

    Para simplificar e cobrir o padrão mais recente (ex: 2021, 2122, 2223, 2324):
    Ano 2020 -> temporada 2020/2021 -> código '2021'
    Ano 2021 -> temporada 2021/2022 -> código '2122'
    Ano 2022 -> temporada 2022/2023 -> código '2223'
    Ano 2023 -> temporada 2023/2024 -> código '2324'
    """
    current_year = datetime.datetime.now().year
    season_codes = []

    for year in range(start_year, current_year + 1):
        # Temporada 2020-2021 é '2021'
        if year == 2020:
            season_codes.append("2021")
        # Temporada 2021-2022 é '2122'
        # Temporada 2022-2023 é '2223'
        # Temporada 2023-2024 é '2324'
        # etc.
        else:
            # Formato YY(YY+1)
            # Ex: para year=2021 (temporada 21/22), code = "2122"
            # Ex: para year=2023 (temporada 23/24), code = "2324"
            start_yy = str(year)[-2:]
            end_yy = str(year + 1)[-2:]
            season_codes.append(f"{start_yy}{end_yy}")

    # Remove duplicados caso a lógica gere algum (improvável com o if/else)
    # e garante que não tentemos buscar temporadas futuras se o script rodar no início do ano.
    # Ex: se hoje é Jan 2024, current_year é 2024.
    # O loop irá até range(2020, 2025)
    # year = 2023 -> 2324 (correto, temporada 23/24 está em andamento ou recém concluída)
    # year = 2024 -> 2425 (pode não existir ainda)
    # A URL para 2425 provavelmente não existirá até ~Julho/Agosto 2024.
    # O tratamento de erro no download cuidará disso.

    # Garante que os códigos sejam únicos e ordenados (embora a lógica atual já deva fazer isso)
    return sorted(list(set(season_codes)))


def download_data(season_code):
    """
    Baixa o arquivo E0.csv para a temporada especificada.
    Salva como /app/E0_{season_code}.csv
    """
    url = BASE_URL.format(season_code=season_code)
    # O nome do arquivo de saída usará o season_code diretamente
    # Ex: para season_code '2021', salva como E0_2021.csv
    # Ex: para season_code '2324', salva como E0_2324.csv
    output_filename = os.path.join(OUTPUT_DIR, f"E0_{season_code}.csv")

    print(f"Tentando baixar dados de: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Lança um erro para códigos HTTP 4xx/5xx

        # Cria o diretório /app/ se não existir (útil para ambientes locais)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print(f"Dados salvos em: {output_filename}")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Erro: Dados não encontrados para a temporada {season_code} (404 Not Found). URL: {url}")
        else:
            print(f"Erro HTTP ao baixar {url}: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Erro de requisição ao baixar {url}: {e}")
    except Exception as e:
        print(f"Erro inesperado ao baixar ou salvar {url}: {e}")
    return False

if __name__ == "__main__":
    print("Iniciando download dos dados de futebol...")
    # O football-data.co.uk usa códigos como "2021" para a temporada 2020-2021,
    # "2122" para 2021-2022, "2223" para 2022-2023, "2324" para 2023-2024.
    # A função get_season_codes foi ajustada para este padrão.

    # Queremos dados desde a temporada 2020/2021.
    # Ano de início para a lógica de temporadas: 2020.
    season_codes_to_download = get_season_codes(start_year=2020)

    print(f"Códigos de temporada a serem baixados: {season_codes_to_download}")

    download_count = 0
    for code in season_codes_to_download:
        if download_data(code):
            download_count += 1

    print(f"Download concluído. {download_count} de {len(season_codes_to_download)} arquivos baixados com sucesso.")

```
