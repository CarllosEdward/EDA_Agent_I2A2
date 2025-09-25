import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import tempfile
from typing import Optional, Dict, Any
from crewai.tools import BaseTool
from pydantic import Field

class ChartGeneratorTool(BaseTool):
    name: str = "Chart Generator"
    description: str = """
    Ferramenta para geração de gráficos e visualizações.
    Cria histogramas, scatter plots, heatmaps, box plots, etc.
    Salva gráficos em arquivos para exibição.
    """
    
    def __init__(self):
        super().__init__()
        # Configurações de estilo
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _run(self, chart_request: str) -> str:
        """
        Gera gráficos baseados na solicitação
        Args:
            chart_request: descrição do gráfico desejado
        """
        try:
            return f"Solicitação de gráfico: {chart_request}"
        except Exception as e:
            return f"Erro ao gerar gráfico: {str(e)}"
    
    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 30) -> str:
        """Cria histograma de uma coluna"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
            ax.set_title(f'Histograma - {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
            
            # Salva em arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except Exception as e:
            plt.close()
            return f"Erro ao criar histograma: {str(e)}"
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           hue_col: Optional[str] = None) -> str:
        """Cria gráfico de dispersão"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if hue_col and hue_col in df.columns:
                # Scatter plot com cores por categoria
                unique_values = df[hue_col].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_values)))
                
                for i, value in enumerate(unique_values):
                    mask = df[hue_col] == value
                    ax.scatter(df[mask][x_col], df[mask][y_col], 
                             c=[colors[i]], label=str(value), alpha=0.7)
                ax.legend()
            else:
                ax.scatter(df[x_col], df[y_col], alpha=0.7)
            
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            ax.grid(True, alpha=0.3)
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except Exception as e:
            plt.close()
            return f"Erro ao criar scatter plot: {str(e)}"
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Cria heatmap de correlação"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .5})
            ax.set_title('Matriz de Correlação')
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except Exception as e:
            plt.close()
            return f"Erro ao criar heatmap: {str(e)}"
    
    def create_box_plot(self, df: pd.DataFrame, column: str, group_by: Optional[str] = None) -> str:
        """Cria box plot"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if group_by and group_by in df.columns:
                df.boxplot(column=column, by=group_by, ax=ax)
            else:
                df.boxplot(column=column, ax=ax)
            
            ax.set_title(f'Box Plot - {column}')
            ax.grid(True, alpha=0.3)
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return temp_file.name
        except Exception as e:
            plt.close()
            return f"Erro ao criar box plot: {str(e)}"

  