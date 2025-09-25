import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool
from pydantic import Field

class DataAnalyzerTool(BaseTool):
    name: str = "Data Analyzer"
    description: str = """
    Ferramenta para análise exploratória de dados CSV.
    Executa análises estatísticas, detecta outliers, calcula correlações.
    Aceita: DataFrame e tipo de análise desejada.
    """

    def _run(self, analysis_request: str) -> str:
        """
        Executa análises nos dados
        Args:
            analysis_request: tipo de análise solicitada
        """
        try:
            # Aqui assumimos que os dados estão em session_state do Streamlit
            # No contexto real, receberemos o DataFrame através dos agentes
            return self._perform_analysis(analysis_request)
            
        except Exception as e:
            return f"Erro na análise: {str(e)}"
    
    def _perform_analysis(self, request: str) -> str:
        """Executa diferentes tipos de análise"""
        # Esta função será expandida pelos agentes
        return f"Análise solicitada: {request}"
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estatísticas básicas do dataset"""
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
        
        # Estatísticas para colunas numéricas
        numeric_stats = {}
        for col in stats_info['column_types']['numeric']:
            numeric_stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'skewness': stats.skew(df[col].dropna()),
                'kurtosis': stats.kurtosis(df[col].dropna())
            }
        
        stats_info['numeric_stats'] = numeric_stats
        return stats_info
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List]:
        """Detecta outliers usando diferentes métodos"""
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
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                threshold = 3
                outliers[col] = df[z_scores > threshold].index.tolist()
        
        return outliers
    
    def calculate_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula matriz de correlação"""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr()
