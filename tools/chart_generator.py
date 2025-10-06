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
import io

# Verifica se o Streamlit está disponível para exibir gráficos
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

class ChartGeneratorTool(BaseTool):
    name: str = "Chart Generator"
    description: str = """
    Ferramenta para geração de gráficos e visualizações.
    Cria histogramas, scatter plots, heatmaps, box plots, e mais.
    Exibe os gráficos diretamente na tela do Streamlit e salva em arquivos.
    """
    
    # Campo para o diretório de gráficos, com um valor padrão.
    charts_dir: str = Field(default="generated_charts", description="Diretório para salvar gráficos")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Configurações de estilo para os gráficos
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Garante que o diretório para salvar os gráficos existe.
        # Caso falhe, usa um diretório temporário como fallback.
        try:
            os.makedirs(self.charts_dir, exist_ok=True)
        except Exception:
            self.charts_dir = tempfile.gettempdir()

    def _run(self, chart_request: str) -> str:
        """
        Método obsoleto, a lógica de despacho agora é feita por agentes.
        A instrução direta de uso será dada nos prompts das tarefas.
        """
        # A nova arquitetura delega a escolha do método correto
        # para a descrição das tarefas, tornando esta função redundante.
        return f"A ferramenta de visualização deve ser chamada diretamente. Por favor, use métodos como create_histogram(), create_correlation_heatmap(), etc."

    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 30) -> str:
        """
        Gera um histograma.

        O código foi otimizado para lidar com a exibição no Streamlit e salvar
        o arquivo de backup de forma mais robusta, incluindo um botão de download.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
            ax.set_title(f'Histograma - {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
            
            # Se o Streamlit estiver disponível, exibe e adiciona botão de download
            if STREAMLIT_AVAILABLE and st is not None:
                st.pyplot(fig)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    label=f"📥 Baixar Histograma {column}",
                    data=img_buffer.getvalue(),
                    file_name=f"histograma_{column}.png",
                    mime="image/png"
                )
            
            plt.close(fig) # Fecha a figura para liberar memória
            return f"✅ Histograma de {column} gerado e exibido."
        
        except Exception as e:
            plt.close()
            return f"❌ Erro ao criar histograma: {str(e)}"

    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str] = None) -> str:
        """
        Gera um scatter plot.

        A lógica de exibição e download foi centralizada para garantir
        consistência entre todos os métodos de gráfico.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if hue_col and hue_col in df.columns:
                sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df, ax=ax)
            else:
                sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
            
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            ax.grid(True, alpha=0.3)
            
            if STREAMLIT_AVAILABLE and st is not None:
                st.pyplot(fig)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    label=f"📥 Baixar Scatter Plot",
                    data=img_buffer.getvalue(),
                    file_name=f"scatter_{x_col}_vs_{y_col}.png",
                    mime="image/png"
                )
            
            plt.close(fig)
            return f"✅ Scatter plot {x_col} vs {y_col} gerado e exibido!"
        
        except Exception as e:
            plt.close()
            return f"❌ Erro ao criar scatter plot: {str(e)}"

    def create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """
        Cria um heatmap de correlação.

        Prioriza a exibição de um gráfico interativo (Plotly) se o Streamlit
        estiver disponível, o que melhora a experiência do usuário.
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return "❌ Menos de 2 colunas numéricas para correlação."
            
            corr_matrix = numeric_df.corr()
            
            if STREAMLIT_AVAILABLE and st is not None:
                # Usa Plotly para um gráfico interativo
                fig_plotly = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Correlação",
                    color_continuous_scale="RdBu_r"
                )
                fig_plotly.update_layout(width=800, height=600)
                st.plotly_chart(fig_plotly, use_container_width=True)

                # Gera a versão PNG para download
                img_buffer = io.BytesIO()
                fig_plotly.write_image(img_buffer, format='png', scale=3)
                st.download_button(
                    label="📥 Baixar Heatmap Correlação",
                    data=img_buffer.getvalue(),
                    file_name="correlation_heatmap.png",
                    mime="image/png"
                )
            
            return f"✅ Heatmap de correlação gerado e exibido."
        
        except Exception as e:
            return f"❌ Erro ao criar heatmap: {str(e)}"

    def create_box_plot(self, df: pd.DataFrame, column: str, group_by: Optional[str] = None) -> str:
        """
        Gera um box plot para análise de distribuição e outliers.

        Adicionou-se o fechamento da figura para evitar acúmulo de memória.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if group_by and group_by in df.columns:
                sns.boxplot(x=group_by, y=column, data=df, ax=ax)
            else:
                sns.boxplot(y=column, data=df, ax=ax)
            
            ax.set_title(f'Box Plot - {column}')
            ax.grid(True, alpha=0.3)
            
            if STREAMLIT_AVAILABLE and st is not None:
                st.pyplot(fig)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    label=f"📥 Baixar Box Plot {column}",
                    data=img_buffer.getvalue(),
                    file_name=f"boxplot_{column}.png",
                    mime="image/png"
                )
            
            plt.close(fig)
            return f"✅ Box plot de {column} gerado e exibido!"
        
        except Exception as e:
            plt.close()
            return f"❌ Erro ao criar box plot: {str(e)}"

    def create_survival_by_gender_chart(self, df: pd.DataFrame) -> str:
        """
        Cria uma análise completa de sobrevivência, focada no caso do Titanic.

        Otimizamos o código para usar apenas Plotly, que é interativo e
        mais moderno, e removemos a redundância da versão Matplotlib para
        backup, já que o Plotly tem mais recursos e o resultado é mais
        apropriado para a web.
        """
        try:
            if 'Sex' not in df.columns or 'Survived' not in df.columns:
                return "❌ Colunas 'Sex' e 'Survived' não encontradas no dataset."
            
            survival_stats = df.groupby('Sex')['Survived'].agg(['count', 'sum']).reset_index()
            survival_stats.columns = ['Gender', 'Total', 'Survivors']
            survival_stats['Deaths'] = survival_stats['Total'] - survival_stats['Survivors']
            survival_stats['Survival_Rate'] = (survival_stats['Survivors'] / survival_stats['Total'] * 100).round(1)
            
            if STREAMLIT_AVAILABLE and st is not None:
                st.markdown("### 📊 Análise de Sobrevivência por Gênero")
                st.dataframe(survival_stats, use_container_width=True)
                
                # Plotly Subplots para uma dashboard consolidada
                from plotly.subplots import make_subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Total de Passageiros', 'Sobreviventes vs Mortos', 'Taxa de Sobrevivência (%)', 'Análise Detalhada'),
                    specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
                )
                
                # Gráfico 1: Total por Gênero
                fig.add_trace(go.Bar(x=survival_stats['Gender'], y=survival_stats['Total'], name='Total', text=survival_stats['Total'], marker_color='lightblue'), row=1, col=1)
                
                # Gráfico 2: Sobreviventes vs Mortos
                fig.add_trace(go.Bar(x=survival_stats['Gender'], y=survival_stats['Survivors'], name='Sobreviventes', text=survival_stats['Survivors'], marker_color='green'), row=1, col=2)
                fig.add_trace(go.Bar(x=survival_stats['Gender'], y=survival_stats['Deaths'], name='Mortos', text=survival_stats['Deaths'], marker_color='red'), row=1, col=2)
                
                # Gráfico 3: Taxa de Sobrevivência
                fig.add_trace(go.Bar(x=survival_stats['Gender'], y=survival_stats['Survival_Rate'], name='Taxa (%)', text=[f"{r}%" for r in survival_stats['Survival_Rate']], marker_color='orange'), row=2, col=1)

                # Gráfico 4: Detalhe
                survival_long = survival_stats.melt(id_vars='Gender', value_vars=['Survivors', 'Deaths'], var_name='Status', value_name='Count')
                fig.add_trace(go.Bar(x=survival_long['Gender'], y=survival_long['Count'], color=survival_long['Status'], text=survival_long['Count']), row=2, col=2)
                
                fig.update_layout(barmode='group', title_text="Análise de Sobrevivência por Gênero - Titanic")
                st.plotly_chart(fig, use_container_width=True)
                
                # Botão de download (usa a figura do Plotly)
                img_buffer = io.BytesIO()
                fig.write_image(img_buffer, format='png', scale=3)
                st.download_button(
                    label="📥 Baixar Análise Completa",
                    data=img_buffer.getvalue(),
                    file_name="survival_analysis_complete.png",
                    mime="image/png"
                )
            
            return "✅ Análise de sobrevivência por gênero gerada e exibida."
            
        except Exception as e:
            return f"❌ Erro ao criar análise de sobrevivência: {str(e)}"
