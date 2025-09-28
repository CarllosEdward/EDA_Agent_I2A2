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

# ADICIONADO: Importações condicionais para Streamlit
try:
    import streamlit as st
    import io
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

class ChartGeneratorTool(BaseTool):
    name: str = "Chart Generator"
    description: str = """
    Ferramenta para geração de gráficos e visualizações.
    Cria histogramas, scatter plots, heatmaps, box plots, etc.
    Salva gráficos em arquivos para exibição e EXIBE NO STREAMLIT quando disponível.
    """
    
    # CORRIGIDO: Declarar charts_dir como campo Pydantic
    charts_dir: str = Field(default="generated_charts", description="Diretório para salvar gráficos")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Configurações de estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Garantir que o diretório existe
        try:
            os.makedirs(self.charts_dir, exist_ok=True)
        except Exception:
            # Fallback para temp directory se não conseguir criar
            self.charts_dir = tempfile.gettempdir()
    
    def _run(self, chart_request: str) -> str:
        """
        Gera gráficos baseados na solicitação
        Args:
            chart_request: descrição do gráfico desejado
        """
        try:
            request_lower = chart_request.lower()
            
            if any(word in request_lower for word in ['sobreviv', 'survival', 'gênero', 'sexo']):
                return self._create_survival_by_gender_info()
            elif any(word in request_lower for word in ['correlação', 'correlation', 'heatmap']):
                return "Use create_correlation_heatmap() para gerar matriz de correlação"
            elif any(word in request_lower for word in ['distribuição', 'histograma']):
                return "Use create_histogram() para gerar distribuição"
            else:
                return f"Solicitação de gráfico identificada: {chart_request}. Use métodos específicos para gerar."
                
        except Exception as e:
            return f"Erro ao processar solicitação: {str(e)}"
    
    def _create_survival_by_gender_info(self) -> str:
        """Informações para gráfico de sobrevivência por gênero"""
        return """
        Para gráfico de sobrevivência por gênero:
        1. Use create_survival_by_gender_chart() 
        2. Verificar se colunas 'Sex' e 'Survived' existem
        3. Gráfico será exibido automaticamente no Streamlit
        """
    
    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 30) -> str:
        """Cria histograma de uma coluna"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
            ax.set_title(f'Histograma - {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
            
            # Exibir no Streamlit se disponível
            if STREAMLIT_AVAILABLE and st is not None:
                st.pyplot(fig)
                
                # Botão de download
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Baixar Histograma {column}",
                    data=img_buffer.getvalue(),
                    file_name=f"histograma_{column}.png",
                    mime="image/png"
                )
            
            # Salvar arquivo de backup
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=self.charts_dir)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ Histograma de {column} gerado e exibido! Arquivo salvo: {temp_file.name}"
            
        except Exception as e:
            plt.close()
            if STREAMLIT_AVAILABLE and st is not None:
                st.error(f"❌ Erro ao criar histograma: {str(e)}")
            return f"Erro ao criar histograma: {str(e)}"
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           hue_col: Optional[str] = None) -> str:
        """Cria gráfico de dispersão"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if hue_col and hue_col in df.columns:
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
            
            # Exibir no Streamlit se disponível
            if STREAMLIT_AVAILABLE and st is not None:
                st.pyplot(fig)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Baixar Scatter Plot {x_col} vs {y_col}",
                    data=img_buffer.getvalue(),
                    file_name=f"scatter_{x_col}_vs_{y_col}.png",
                    mime="image/png"
                )
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=self.charts_dir)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ Scatter plot {x_col} vs {y_col} gerado e exibido! Arquivo: {temp_file.name}"
            
        except Exception as e:
            plt.close()
            if STREAMLIT_AVAILABLE and st is not None:
                st.error(f"❌ Erro ao criar scatter plot: {str(e)}")
            return f"Erro ao criar scatter plot: {str(e)}"
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Cria heatmap de correlação"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                msg = "❌ Menos de 2 colunas numéricas para correlação"
                if STREAMLIT_AVAILABLE and st is not None:
                    st.warning(msg)
                return msg
            
            corr_matrix = numeric_df.corr()
            
            # Versão Plotly interativa (se Streamlit disponível)
            if STREAMLIT_AVAILABLE and st is not None:
                fig_plotly = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Correlação",
                    color_continuous_scale="RdBu_r"
                )
                fig_plotly.update_layout(width=800, height=600)
                st.plotly_chart(fig_plotly, use_container_width=True)
            
            # Versão matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .5})
            ax.set_title('Matriz de Correlação')
            
            if STREAMLIT_AVAILABLE and st is not None:
                st.pyplot(fig)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Baixar Heatmap Correlação",
                    data=img_buffer.getvalue(),
                    file_name="correlation_heatmap.png",
                    mime="image/png"
                )
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=self.charts_dir)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ Heatmap de correlação gerado e exibido! Arquivo: {temp_file.name}"
            
        except Exception as e:
            plt.close()
            if STREAMLIT_AVAILABLE and st is not None:
                st.error(f"❌ Erro ao criar heatmap: {str(e)}")
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
            
            if STREAMLIT_AVAILABLE and st is not None:
                st.pyplot(fig)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Baixar Box Plot {column}",
                    data=img_buffer.getvalue(),
                    file_name=f"boxplot_{column}.png",
                    mime="image/png"
                )
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=self.charts_dir)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ Box plot de {column} gerado e exibido! Arquivo: {temp_file.name}"
            
        except Exception as e:
            plt.close()
            if STREAMLIT_AVAILABLE and st is not None:
                st.error(f"❌ Erro ao criar box plot: {str(e)}")
            return f"Erro ao criar box plot: {str(e)}"
    
    def create_survival_by_gender_chart(self, df: pd.DataFrame) -> str:
        """Cria gráfico completo de sobrevivência por gênero"""
        try:
            # Verificar se colunas existem
            if 'Sex' not in df.columns or 'Survived' not in df.columns:
                msg = "❌ Colunas 'Sex' e 'Survived' não encontradas no dataset"
                if STREAMLIT_AVAILABLE and st is not None:
                    st.error(msg)
                return msg
            
            # Análise dos dados
            survival_stats = df.groupby('Sex')['Survived'].agg(['count', 'sum']).reset_index()
            survival_stats.columns = ['Gender', 'Total', 'Survivors']
            survival_stats['Deaths'] = survival_stats['Total'] - survival_stats['Survivors']
            survival_stats['Survival_Rate'] = (survival_stats['Survivors'] / survival_stats['Total'] * 100).round(1)
            
            if STREAMLIT_AVAILABLE and st is not None:
                st.markdown("### 📊 Análise de Sobrevivência por Gênero")
                st.dataframe(survival_stats, use_container_width=True)
                
                # Gráficos Plotly interativos
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Total de Passageiros por Gênero',
                        'Sobreviventes vs Não Sobreviventes', 
                        'Taxa de Sobrevivência (%)',
                        'Comparação Detalhada'
                    ),
                    specs=[[{"type": "bar"}, {"type": "bar"}],
                           [{"type": "bar"}, {"type": "bar"}]]
                )
                
                # Adicionar traces
                fig.add_trace(
                    go.Bar(x=survival_stats['Gender'], y=survival_stats['Total'], 
                           name='Total', marker_color='lightblue', text=survival_stats['Total']),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=survival_stats['Gender'], y=survival_stats['Survivors'], 
                           name='Sobreviventes', marker_color='green', text=survival_stats['Survivors']),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Bar(x=survival_stats['Gender'], y=survival_stats['Deaths'], 
                           name='Não Sobreviventes', marker_color='red', text=survival_stats['Deaths']),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(x=survival_stats['Gender'], y=survival_stats['Survival_Rate'], 
                           name='Taxa (%)', marker_color='orange', 
                           text=[f"{rate}%" for rate in survival_stats['Survival_Rate']]),
                    row=2, col=1
                )
                
                categories = []
                values = []
                colors = []
                for _, row in survival_stats.iterrows():
                    categories.extend([f"{row['Gender']} - Total", f"{row['Gender']} - Sobrev."])
                    values.extend([row['Total'], row['Survivors']])
                    colors.extend(['lightblue', 'green'])
                
                fig.add_trace(
                    go.Bar(x=categories, y=values, marker_color=colors, text=values),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, title_text="Análise de Sobrevivência por Gênero - Titanic")
                st.plotly_chart(fig, use_container_width=True)
            
            # Versão matplotlib para backup
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            bars1 = plt.bar(survival_stats['Gender'], survival_stats['Total'], color='lightblue', alpha=0.7)
            plt.title('Total de Passageiros por Gênero')
            plt.ylabel('Número de Passageiros')
            for bar, value in zip(bars1, survival_stats['Total']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        str(value), ha='center', va='bottom', fontweight='bold')
            
            plt.subplot(2, 2, 2)
            width = 0.35
            x = np.arange(len(survival_stats['Gender']))
            bars2 = plt.bar(x - width/2, survival_stats['Survivors'], width, label='Sobreviventes', color='green', alpha=0.7)
            bars3 = plt.bar(x + width/2, survival_stats['Deaths'], width, label='Não Sobreviventes', color='red', alpha=0.7)
            plt.title('Sobreviventes vs Não Sobreviventes')
            plt.ylabel('Número de Pessoas')
            plt.xticks(x, survival_stats['Gender'])
            plt.legend()
            
            plt.subplot(2, 2, 3)
            bars4 = plt.bar(survival_stats['Gender'], survival_stats['Survival_Rate'], color='orange', alpha=0.7)
            plt.title('Taxa de Sobrevivência por Gênero')
            plt.ylabel('Taxa de Sobrevivência (%)')
            for bar, rate in zip(bars4, survival_stats['Survival_Rate']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{rate}%', ha='center', va='bottom', fontweight='bold')
            
            plt.subplot(2, 2, 4)
            table_data = []
            for _, row in survival_stats.iterrows():
                table_data.append([row['Gender'], f"{row['Total']}", f"{row['Survivors']}", 
                                 f"{row['Deaths']}", f"{row['Survival_Rate']}%"])
            
            table = plt.table(cellText=table_data,
                            colLabels=['Gênero', 'Total', 'Sobrev.', 'Mortos', 'Taxa (%)'],
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            plt.axis('off')
            plt.title('Resumo Estatístico')
            
            plt.tight_layout()
            
            if STREAMLIT_AVAILABLE and st is not None:
                st.pyplot(plt)
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Baixar Análise Completa PNG",
                    data=img_buffer.getvalue(),
                    file_name="survival_by_gender_complete.png",
                    mime="image/png"
                )
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=self.charts_dir)
            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            return f"✅ Análise completa de sobrevivência por gênero gerada! Gráficos exibidos acima. Arquivo: {temp_file.name}"
            
        except Exception as e:
            plt.close()
            if STREAMLIT_AVAILABLE and st is not None:
                st.error(f"❌ Erro ao criar análise de sobrevivência: {str(e)}")
            return f"Erro ao criar análise de sobrevivência: {str(e)}"
