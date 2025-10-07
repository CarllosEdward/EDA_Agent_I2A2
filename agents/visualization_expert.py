from crewai import Agent
from tools import ChartGeneratorTool, DataAnalyzerTool, MemoryManagerTool
import streamlit as st  # ADICIONADO: Integração com Streamlit
import pandas as pd  # ADICIONADO: Para manipulação de dados

def create_visualization_expert_agent(llm):
    """Cria o agente especialista em visualização com prompts melhorados."""
    
    return Agent(
        role="Especialista em Visualização de Dados",
                goal="""
                Produzir visualizações que expressem claramente os padrões e insights do
                dataset. Prefira gráficos que favoreçam a comunicação rápida e correta.
                """,
                backstory="""
                Como especialista em visualização, sua tarefa é transformar dados e análises
                em representações visuais que permitam decisões rápidas.

                Responsabilidades:
                - Escolher o tipo de gráfico adequado (histograma, heatmap, comparação etc.).
                - Gerar e exibir a visualização usando as ferramentas disponíveis.
                - Fornecer um breve resumo dos pontos relevantes que o gráfico revela.

                Priorizamos a clareza e a utilidade: cada gráfico deve responder a uma
                pergunta concreta do usuário.
                """,
        tools=[
            ChartGeneratorTool(),
            DataAnalyzerTool(),
            MemoryManagerTool()
        ],
        llm=llm,
        verbose=True,
        memory=True,
        allow_delegation=False,
        max_iter=2,
        max_execution_time=90,
        system_message="""
        Quando solicitado, gere a visualização mais informativa e explique brevemente
        o insight principal. Utilize o ChartGeneratorTool para exibir e permitir
        o download do gráfico.
        """
    )

# NOVA CLASSE: Wrapper para métodos diretos de visualização
class VisualizationExpert:
    """
    Classe wrapper para facilitar chamadas diretas de visualização
    """
    
    def __init__(self, llm):
        self.agent = create_visualization_expert_agent(llm)
        self.chart_tool = ChartGeneratorTool()
    
    def create_survival_chart_direct(self, data: pd.DataFrame, user_question: str = "") -> str:
        """
        Método direto para criar gráfico de sobrevivência
        """
        try:
            st.info("🎨 Gerando gráfico de sobrevivência...")
            
            # Verificar se é dataset do Titanic ou similar
            if 'Sex' in data.columns and 'Survived' in data.columns:
                # Usar método específico para sobrevivência por gênero
                result = self.chart_tool.create_survival_by_gender_chart(data)
                
                # Análise adicional dos dados
                total_passengers = len(data)
                total_survivors = data['Survived'].sum()
                survival_rate = (total_survivors / total_passengers * 100)
                
                analysis = f"""
                **📊 Análise de Sobrevivência Completa:**
                
                • **Total de passageiros**: {total_passengers}
                • **Total de sobreviventes**: {total_survivors}
                • **Taxa geral de sobrevivência**: {survival_rate:.1f}%
                
                **🔍 Insights por Gênero:**
                """
                
                # Análise por gênero
                gender_analysis = data.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
                for gender, stats in gender_analysis.iterrows():
                    gender_pt = "Mulheres" if gender == "female" else "Homens"
                    emoji = "👩" if gender == "female" else "👨"
                    analysis += f"""
                • {emoji} **{gender_pt}**: {stats['sum']}/{stats['count']} sobreviveram ({stats['mean']*100:.1f}%)"""
                
                analysis += f"""
                
                **✅ Gráficos gerados acima mostram:**
                - Comparação de totais por gênero
                - Distribuição de sobreviventes vs não sobreviventes  
                - Taxas de sobrevivência percentuais
                - Resumo estatístico completo
                
                {result}
                """
                
                return analysis
                
            else:
                # Dataset sem colunas de sobrevivência padrão
                st.warning("⚠️ Colunas 'Sex' e 'Survived' não encontradas. Criando análise geral...")
                
                # Tentar criar visualização geral
                if len(data.select_dtypes(include=['number']).columns) > 0:
                    return self.create_general_visualization(data)
                else:
                    return "❌ Dataset não contém dados adequados para visualização de sobrevivência."
                    
        except Exception as e:
            error_msg = f"Erro ao criar gráfico de sobrevivência: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def create_correlation_chart_direct(self, data: pd.DataFrame) -> str:
        """
        Método direto para criar matriz de correlação
        """
        try:
            st.info("🎨 Gerando matriz de correlação...")
            
            # Verificar se há colunas numéricas suficientes
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                msg = "⚠️ Dataset possui menos de 2 colunas numéricas para análise de correlação."
                st.warning(msg)
                return msg
            
            result = self.chart_tool.create_correlation_heatmap(data)
            
            # Análise das correlações
            corr_matrix = data[numeric_cols].corr()
            
            # Encontrar correlações mais fortes (excluindo diagonal)
            import numpy as np
            corr_abs = corr_matrix.abs()
            np.fill_diagonal(corr_abs.values, 0)
            
            if not corr_abs.empty and corr_abs.max().max() > 0:
                max_corr = corr_abs.max().max()
                max_corr_idx = corr_abs.stack().idxmax()
                actual_corr = corr_matrix.loc[max_corr_idx[0], max_corr_idx[1]]
                
                analysis = f"""
                **📊 Análise de Correlação:**
                
                • **Colunas analisadas**: {len(numeric_cols)} variáveis numéricas
                • **Correlação mais forte**: {max_corr_idx[0]} ↔ {max_corr_idx[1]} ({actual_corr:.3f})
                
                **🔍 Interpretação:**
                • Valores próximos de +1: Correlação positiva forte
                • Valores próximos de -1: Correlação negativa forte
                • Valores próximos de 0: Pouca ou nenhuma correlação
                
                **✅ O heatmap acima mostra todas as correlações entre as variáveis numéricas.**
                
                {result}
                """
            else:
                analysis = f"""
                **📊 Matriz de Correlação Gerada:**
                
                • **Colunas analisadas**: {len(numeric_cols)} variáveis numéricas
                • **Correlações**: Visualizadas no heatmap acima
                
                {result}
                """
            
            return analysis
            
        except Exception as e:
            error_msg = f"Erro ao criar matriz de correlação: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def create_distribution_chart_direct(self, data: pd.DataFrame, column: str = None) -> str:
        """
        Método direto para criar gráfico de distribuição
        """
        try:
            # Selecionar coluna se não especificada
            if column is None:
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    column = numeric_cols[0]
                else:
                    msg = "❌ Nenhuma coluna numérica encontrada para análise de distribuição."
                    st.warning(msg)
                    return msg
            
            if column not in data.columns:
                msg = f"❌ Coluna '{column}' não encontrada no dataset."
                st.error(msg)
                return msg
            
            st.info(f"🎨 Gerando gráfico de distribuição para '{column}'...")
            
            result = self.chart_tool.create_histogram(data, column)
            
            # Análise estatística da distribuição
            stats = data[column].describe()
            null_count = data[column].isnull().sum()
            
            analysis = f"""
            **📊 Análise de Distribuição - {column}:**
            
            **📈 Estatísticas Descritivas:**
            • Média: {stats['mean']:.2f}
            • Mediana (50%): {stats['50%']:.2f}
            • Desvio Padrão: {stats['std']:.2f}
            • Mínimo: {stats['min']:.2f}
            • Máximo: {stats['max']:.2f}
            • 1º Quartil (25%): {stats['25%']:.2f}
            • 3º Quartil (75%): {stats['75%']:.2f}
            • Valores ausentes: {null_count}
            
            **✅ O histograma acima mostra a distribuição da variável {column}.**
            
            {result}
            """
            
            return analysis
            
        except Exception as e:
            error_msg = f"Erro ao criar gráfico de distribuição: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def create_general_visualization(self, data: pd.DataFrame) -> str:
        """
        Cria visualização geral baseada na estrutura do dataset
        """
        try:
            st.info("🎨 Gerando visualizações gerais do dataset...")
            
            results = []
            
            # 1. Se há colunas numéricas, criar correlação
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                corr_result = self.create_correlation_chart_direct(data)
                results.append("🔗 **Matriz de Correlação gerada**")
            
            # 2. Distribuição da primeira coluna numérica
            if len(numeric_cols) > 0:
                first_numeric = numeric_cols[0]
                dist_result = self.create_distribution_chart_direct(data, first_numeric)
                results.append(f"📊 **Distribuição de {first_numeric} gerada**")
            
            # 3. Informações gerais
            general_info = f"""
            **📋 Informações Gerais do Dataset:**
            
            • **Dimensões**: {data.shape[0]} linhas × {data.shape[1]} colunas
            • **Colunas numéricas**: {len(numeric_cols)}
            • **Colunas categóricas**: {len(data.select_dtypes(include=['object']).columns)}
            • **Valores ausentes**: {data.isnull().sum().sum()}
            
            **✅ Visualizações geradas:**
            {chr(10).join(results)}
            """
            
            return general_info
            
        except Exception as e:
            error_msg = f"Erro ao criar visualização geral: {str(e)}"
            st.error(error_msg)
            return error_msg
