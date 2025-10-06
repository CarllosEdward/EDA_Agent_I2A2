import pandas as pd
import os
import requests
import tempfile
from typing import Union, Tuple
import streamlit as st
import time

# A importação da classe de configuração está aqui para fins de demonstração.
# Presume-se que a estrutura do projeto já inclua o módulo 'utils'.
from utils.config import Config

def ensure_directories():
    """
    Cria os diretórios necessários para o projeto se eles ainda não existirem.
    Isso é crucial para garantir que operações de leitura/escrita, como
    salvar arquivos temporários ou gráficos gerados, não falhem.
    """
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

def validate_csv_file(file_path: str) -> Tuple[bool, str]:
    """
    Verifica se um arquivo CSV é válido e pode ser lido pelo pandas.
    Lê apenas as primeiras 5 linhas para uma validação rápida e eficiente,
    evitando carregar arquivos grandes desnecessariamente.
    Retorna uma tupla com um booleano (True se válido) e uma mensagem de status.
    """
    try:
        df = pd.read_csv(file_path, nrows=5)
        if df.empty:
            return False, "Arquivo CSV está vazio."
        if len(df.columns) == 0:
            return False, "Arquivo CSV não possui colunas."
        return True, f"Arquivo válido com {len(df.columns)} colunas."
    except Exception as e:
        return False, f"Erro ao ler arquivo: {str(e)}"

def download_csv_from_url(url: str) -> str:
    """
    Baixa um arquivo CSV de uma URL e o salva em um arquivo temporário.
    É fundamental para o manuseio de dados da web. Utiliza 'tempfile' para
    garantir que os arquivos sejam criados e gerenciados de forma segura,
    sem a necessidade de um caminho fixo. Lança uma exceção em caso de falha.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Lança exceção para status de erro (4xx ou 5xx).
        
        # Cria um arquivo temporário para armazenar o conteúdo.
        temp_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False)
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        raise Exception(f"Erro ao baixar CSV: {str(e)}")

def clean_temp_files():
    """
    Limpa arquivos temporários antigos do diretório de uploads.
    Essa rotina de manutenção é importante para evitar que o armazenamento
    seja sobrecarregado com arquivos desnecessários de sessões anteriores.
    Remove arquivos com mais de 1 hora de idade.
    """
    temp_dir = Config.UPLOAD_DIR
    if not os.path.exists(temp_dir):
        return
    
    current_time = time.time()
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            # Compara o tempo atual com o tempo de modificação do arquivo.
            if current_time - os.path.getmtime(file_path) > 3600:
                try:
                    os.remove(file_path)
                except Exception:
                    # Ignora erros de permissão ou arquivos em uso.
                    pass

def format_number(num: float, decimals: int = 2) -> str:
    """
    Formata um número para uma string mais legível, usando sufixos como 'K' para
    milhares e 'M' para milhões. Isso melhora a apresentação de dados em relatórios
    e visualizações, tornando-os mais fáceis de entender.
    """
    if abs(num) >= 1e6:
        return f"{num/1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"

def get_column_types(df: pd.DataFrame) -> dict:
    """
    Analisa os tipos de dados das colunas de um DataFrame e os categoriza.
    Essa função é fundamental para a análise exploratória, pois permite que
    os agentes saibam quais colunas são numéricas, categóricas ou de data,
    direcionando as análises subsequentes.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }
