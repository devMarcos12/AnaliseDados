import os
import requests
import zipfile
import io
import pandas as pd


def main():
    print('Iniciando pipeline de dados')
    
    token = 'KGAT_b34cc39da30e4724726445f93a515e89'

    URL = "https://www.kaggle.com/api/v1/datasets/download/adaoduque/campeonato-brasileiro-de-futebol"
    headers = {"Authorization": f"Bearer {token}"}

    print("Baixando os arquivos do dataset...")
    response = requests.get(URL, headers=headers, stream=True)

    if response.status_code == 200:
        print("Download concluído! Extraindo arquivos...")

        download_path = './dados_brasileirao'
        os.makedirs(download_path, exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(download_path)
        
        df_partidas = pd.read_csv(f'{download_path}/campeonato-brasileiro-full.csv')
        df_cartoes = pd.read_csv(f'{download_path}/campeonato-brasileiro-cartoes.csv')
        
        df_integrado = pd.merge(
            df_partidas, 
            df_cartoes, 
            left_on='ID', 
            right_on='partida_id', 
            how='inner'
        )
        
        caminho_arquivo_final = f'{download_path}/base_integrada_projeto.csv'
        df_integrado.to_csv(caminho_arquivo_final, index=False)
        
    else:
        print(response.status_code)
        print(response.text)

if __name__ == '__main__':
    main()