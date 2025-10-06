import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool
from pydantic import Field

class DataAnalyzerTool(BaseTool):
    name: str = "Data Analyzer"
    description: str = """
    Ferramenta para análise exploratória de dados. Executa estatísticas descritivas,
    detecta outliers, e calcula a matriz de correlação.
    """

    # O método _run é obsoleto e a chamada de métodos deve ser feita
    # diretamente pelos agentes, conforme a nova arquitetura do CrewAI.
    def _run(self, request: str) -> str:
        return "A ferramenta 'DataAnalyzer' deve ser chamada diretamente com seus métodos específicos, como get_basic_stats(), detect_outliers() ou calculate_correlations()."
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Retorna um dicionário com estatísticas básicas do dataset.
        
        Isso inclui o formato (linhas e colunas), tipos de dados de cada coluna,
        contagem de valores ausentes e estatísticas descritivas para as colunas numéricas.
        É a base para a análise inicial de qualquer dataset.
        """
        stats_info = {
            'shape': df.shape,
            'column_types': {
                'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime': df.select_dtypes(include=['datetime']).columns.tolist()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicated_rows': df.duplicated().sum()
        }
        
        # Apenas calcula estatísticas numéricas se houver colunas numéricas
        numeric_stats = {}
        for col in stats_info['column_types']['numeric']:
            if not df[col].empty:
                numeric_stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75),
                    'skewness': stats.skew(df[col].dropna()) if df[col].dropna().shape[0] > 0 else np.nan,
                    'kurtosis': stats.kurtosis(df[col].dropna()) if df[col].dropna().shape[0] > 0 else np.nan
                }
        
        stats_info['numeric_stats'] = numeric_stats
        return stats_info
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List]:
        """
        Detecta outliers em colunas numéricas usando o método IQR ou Z-score.
        
        Essa função é fundamental para a fase de pré-processamento,
        ajudando a identificar valores atípicos que podem distorcer a análise estatística
        e os resultados de modelos de machine learning.
        """
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                # Evita erro em colunas vazias
                if not df[col].dropna().empty:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    threshold = 3
                    outliers[col] = df[z_scores > threshold].index.tolist()
                else:
                    outliers[col] = []
        
        return outliers
    
    def calculate_correlations(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calcula a matriz de correlação para todas as colunas numéricas.
        
        Essa matriz é essencial para entender a relação linear entre as variáveis,
        sendo um passo importante na análise exploratória de dados.
        Retorna None se não houver colunas numéricas suficientes.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
            
        return numeric_df.corr()
