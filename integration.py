import pandas as pd
import os

def IntegrateData():
    print("Iniciando a integração dos dados...")
    
    # IMPORTANTE: Troque os nomes abaixo para o nome exato dos arquivos .gz que o Chrome baixou
    arquivo_serie_a = 'mundo_transfermarkt_competicoes_brasileirao_serie_a.csv.gz'
    arquivo_copa = 'mundo_transfermarkt_competicoes_copa_brasil.csv.gz'
    
    # O Pandas descompacta e lê o arquivo em memória automaticamente
    df_br = pd.read_csv(arquivo_serie_a, compression='gzip')
    df_copa = pd.read_csv(arquivo_copa, compression='gzip')
    
    # Criando uma coluna para identificar a origem de cada linha após juntar tudo
    df_br['campeonato_origem'] = 'Brasileirão Série A'
    df_copa['campeonato_origem'] = 'Copa do Brasil'
    
    # Empilhando as duas tabelas
    df_integrado = pd.concat([df_br, df_copa], ignore_index=True)
    
    print(f"Total de registros na base integrada: {len(df_integrado)}")
    
    # Exportando o resultado final (já descompactado e unificado) para o Ramalho usar
    nome_arquivo_final = 'base_futebol_integrada.csv'
    df_integrado.to_csv(nome_arquivo_final, index=False)
    
    print(f"Integração concluída! Arquivo {nome_arquivo_final} gerado com sucesso.")

if __name__ == "__main__":
    IntegrateData()