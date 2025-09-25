import pandas as pd
import requests
import os
import tempfile
from typing import Union, Optional, Tuple
from crewai.tools import BaseTool
from pydantic import Field
from utils.helpers import validate_csv_file, download_csv_from_url

class CSVLoaderTool(BaseTool):
    name: str = "CSV Loader"
    description: str = """
    Ferramenta para carregar arquivos CSV de arquivos locais ou URLs.
    Aceita: caminho do arquivo, URL, ou dados já carregados.
    Retorna: DataFrame pandas carregado e validado.
    """

    def _run(self, file_source: str) -> str:
        """
        Carrega CSV de arquivo local ou URL
        Args:
            file_source: caminho do arquivo ou URL
        Returns:
            String com informações sobre o DataFrame carregado
        """
        try:
            # Verifica se é URL
            if file_source.startswith(('http://', 'https://')):
                # Download da URL
                temp_path = download_csv_from_url(file_source)
                df = pd.read_csv(temp_path)
                # Remove arquivo temporário
                os.unlink(temp_path)
            else:
                # Arquivo local
                is_valid, message = validate_csv_file(file_source)
                if not is_valid:
                    return f"Erro na validação: {message}"
                df = pd.read_csv(file_source)

            # Informações básicas do DataFrame
            info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'has_nulls': df.isnull().any().any(),
                'null_counts': df.isnull().sum().to_dict()
            }
            
            return f"""
            CSV carregado com sucesso:
            - Linhas: {info['rows']}
            - Colunas: {info['columns']}
            - Nomes das colunas: {', '.join(info['column_names'][:10])}{'...' if len(info['column_names']) > 10 else ''}
            - Uso de memória: {info['memory_usage'] / (1024*1024):.2f} MB
            - Possui valores nulos: {'Sim' if info['has_nulls'] else 'Não'}
            - Primeiras 3 linhas como amostra:
            {df.head(3).to_string()}
            """
            
        except Exception as e:
            return f"Erro ao carregar CSV: {str(e)}"
