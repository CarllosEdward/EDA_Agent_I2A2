import pandas as pd
import requests
import os
import tempfile
from typing import Union, Optional, Tuple
from crewai.tools import BaseTool
from pydantic import Field
# A importação das funções de validação e download sugere uma
# estrutura de projeto maior, que ajuda a manter o código limpo.
from utils.helpers import validate_csv_file, download_csv_from_url

class CSVLoaderTool(BaseTool):
    name: str = "CSV Loader"
    description: str = """
    Ferramenta para carregar arquivos CSV de caminhos locais ou URLs.
    Retorna o DataFrame carregado junto com um resumo das suas características.
    """
    
    def _run(self, file_source: str) -> str:
        """
        Carrega um arquivo CSV, valida e retorna um resumo.
        
        Essa função serve como a principal interface para os agentes. Ela automatiza
        o processo de identificar se a fonte é um arquivo local ou uma URL e trata
        a leitura de forma transparente, facilitando o trabalho do agente.
        
        Args:
            file_source: O caminho do arquivo ou a URL para o arquivo CSV.
            
        Returns:
            Uma string com um relatório conciso sobre o status do carregamento
            e as características principais do DataFrame, como dimensões,
            nomes de colunas e valores nulos.
        """
        try:
            # Detecta se a fonte é uma URL e faz o download para um arquivo temporário.
            if file_source.startswith(('http://', 'https://')):
                temp_path = download_csv_from_url(file_source)
                df = pd.read_csv(temp_path)
                os.unlink(temp_path) # Limpa o arquivo temporário após o uso.
            else:
                # Trata arquivos locais, validando antes de tentar ler para evitar erros.
                is_valid, message = validate_csv_file(file_source)
                if not is_valid:
                    return f"Erro na validação: {message}"
                df = pd.read_csv(file_source)

            # Gera um dicionário com informações essenciais do DataFrame.
            info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
                'has_nulls': df.isnull().any().any(),
                'null_counts': df.isnull().sum().to_dict()
            }
            
            # Formata a saída para ser fácil de ler para o agente e para o usuário.
            # O to_string() é usado para uma visualização clara das primeiras linhas.
            return f"""
            CSV carregado com sucesso!
            ---
            - **Linhas**: {info['rows']}
            - **Colunas**: {info['columns']}
            - **Nomes das colunas**: {', '.join(info['column_names'][:10])}{'...' if len(info['column_names']) > 10 else ''}
            - **Valores nulos**: {'Sim' if info['has_nulls'] else 'Não'}
            ---
            **Amostra de Dados:**
            {df.head(3).to_string()}
            """
        
        except Exception as e:
            return f"❌ Erro ao carregar CSV: {str(e)}"
