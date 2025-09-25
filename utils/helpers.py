import pandas as pd
import os
import requests
import tempfile
from typing import Union, Tuple
import streamlit as st

def ensure_directories():
    """Cria diretórios necessários se não existirem"""
    from utils.config import Config
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

def validate_csv_file(file_path: str) -> Tuple[bool, str]:
    """
    Valida se o arquivo CSV pode ser lido
    Returns: (is_valid, message)
    """
    try:
        df = pd.read_csv(file_path, nrows=5)  # Lê apenas 5 linhas para validação
        if df.empty:
            return False, "Arquivo CSV está vazio"
        if len(df.columns) == 0:
            return False, "Arquivo CSV não possui colunas"
        return True, f"Arquivo válido com {len(df.columns)} colunas"
    except Exception as e:
        return False, f"Erro ao ler arquivo: {str(e)}"

def download_csv_from_url(url: str) -> str:
    """
    Download CSV de URL e salva temporariamente
    Returns: caminho do arquivo baixado
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Cria arquivo temporário
        temp_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False)
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        raise Exception(f"Erro ao baixar CSV: {str(e)}")

def clean_temp_files():
    """Remove arquivos temporários antigos"""
    from utils.config import Config
    import time
    
    temp_dir = Config.UPLOAD_DIR
    if not os.path.exists(temp_dir):
        return
    
    current_time = time.time()
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            # Remove arquivos mais antigos que 1 hora
            if current_time - os.path.getmtime(file_path) > 3600:
                try:
                    os.remove(file_path)
                except:
                    pass

def format_number(num: float, decimals: int = 2) -> str:
    """Formata números para exibição"""
    if abs(num) >= 1e6:
        return f"{num/1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"

def get_column_types(df: pd.DataFrame) -> dict:
    """Retorna tipos de colunas categorizados"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }
