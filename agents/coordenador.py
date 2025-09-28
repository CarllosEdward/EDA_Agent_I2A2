from crewai import Agent
from tools import CSVLoaderTool, DataAnalyzerTool, ChartGeneratorTool, MemoryManagerTool
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

def create_coordenador_agent(llm):
    """Cria o agente coordenador principal (evoluído com detecção de visualização)"""
    
    return Agent(
        role="Coordenador de Análise de Dados",
        goal="""
        Coordenar e supervisionar todo o processo de análise exploratória de dados CSV.
        Gerenciar o fluxo de trabalho entre os agentes especializados e manter a memória
        das análises realizadas para fornecer conclusões consolidadas.
        
        NOVO: Detectar automaticamente quando o usuário solicita visualizações e
        coordenar com o agente de visualização para gerar gráficos que apareçam na tela.
        """,
        backstory="""
        Você é um cientista de dados experiente com expertise em análise exploratória.
        Sua função é orquestrar o trabalho da equipe, delegando tarefas específicas aos
        agentes especializados e consolidando os resultados em respostas claras e úteis.
        
        Você tem memória das análises anteriores e pode fornecer insights baseados no
        histórico completo da sessão de análise. Sempre mantenha o foco na pergunta
        do usuário e direcione a equipe para fornecer a melhor resposta possível.
        
        NOVA CAPACIDADE: Você agora identifica automaticamente solicitações de visualização
        e coordena a geração de gráficos que aparecem diretamente na tela do Streamlit.
        
        IMPORTANTE: Seja conciso mas compreensivo em suas respostas. Evite repetições
        desnecessárias e foque nos insights mais relevantes. Quando não for explicitamente
        solicitado detalhes extensos, mantenha respostas objetivas e diretas.
        """,
        tools=[
            CSVLoaderTool(),
            MemoryManagerTool(),
            DataAnalyzerTool(),
            ChartGeneratorTool()
        ],
        llm=llm,
        verbose=True,
        memory=True,
        allow_delegation=True,
        max_iter=2,  # Mantido: Reduzido de 3 para 2
        max_execution_time=120,  # Mantido: Limite de 2 minutos
        system_message="""Priorize eficiência: respostas concisas mas compreensivas. 
        Evite análises redundantes. Foque no essencial quando não solicitado detalhamento.
        SEMPRE gere gráficos quando solicitado e certifique-se que apareçam na tela."""
    )

# NOVA CLASSE: Coordenador inteligente com detecção automática
class CoordenadorInteligente:
    """
    Coordenador evoluído que detecta automaticamente necessidades de visualização
    """
    
    def __init__(self, llm):
        self.agent = create_coordenador_agent(llm)
        self.chart_tool = ChartGeneratorTool()
    
    def analyze_user_request(self, user_question: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa a pergunta do usuário e determina estratégia de resposta
        """
        question_lower = user_question.lower()
        
        analysis_plan = {
            'needs_visualization': False,
            'needs_statistical_analysis': True,  # Sempre incluir alguma análise
            'visualization_type': None,
            'priority': 'analysis',  # 'analysis', 'visualization', or 'both'
            'specific_columns': [],
            'response_strategy': 'standard'
        }
        
        # NOVO: Detecção inteligente de palavras-chave
        viz_keywords = {
            'explicit_chart': ['gráfico', 'grafico', 'chart', 'plot', 'visualiza', 'mostra gráfico', 'gera gráfico'],
            'survival': ['sobreviv', 'survival', 'morreu', 'morte', 'vivo', 'titanic'],
            'correlation': ['correlação', 'correlacao', 'correlation', 'relação', 'relacionamento', 'heatmap'],
            'distribution': ['distribuição', 'distribuicao', 'distribution', 'histograma', 'frequency'],
            'comparison': ['compar', 'diferença', 'vs', 'versus', 'entre', 'separando']
        }
        
        # Detectar necessidade de visualização
        for category, keywords in viz_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                analysis_plan['needs_visualization'] = True
                
                if category == 'explicit_chart':
                    analysis_plan['priority'] = 'visualization'
                elif category == 'survival':
                    analysis_plan['visualization_type'] = 'survival'
                elif category == 'correlation':
                    analysis_plan['visualization_type'] = 'correlation'
                elif category == 'distribution':
                    analysis_plan['visualization_type'] = 'distribution'
                
                break
        
        # NOVO: Detecção de colunas específicas mencionadas
        for col in data.columns:
            if col.lower() in question_lower:
                analysis_plan['specific_columns'].append(col)
        
        # NOVO: Detecção especial para Titanic
        if ('gênero' in question_lower or 'sexo' in question_lower or 'homens' in question_lower or 'mulheres' in question_lower) and \
           ('sobreviv' in question_lower or 'Sex' in data.columns):
            analysis_plan['visualization_type'] = 'survival_by_gender'
            analysis_plan['needs_visualization'] = True
            analysis_plan['priority'] = 'visualization'
        
        # Determinar estratégia de resposta
        if analysis_plan['needs_visualization'] and analysis_plan['needs_statistical_analysis']:
            analysis_plan['response_strategy'] = 'combined'
        elif analysis_plan['needs_visualization']:
            analysis_plan['response_strategy'] = 'visualization_focused'
        else:
            analysis_plan['response_strategy'] = 'analysis_focused'
        
        return analysis_plan
    
    def coordinate_response(self, user_question: str, data: pd.DataFrame, 
                          data_explorer_agent=None, visualization_expert=None) -> str:
        """
        Coordena resposta baseada na análise da pergunta
        """
        try:
            # Analisar solicitação
            plan = self.analyze_user_request(user_question, data)
            
            # Feedback visual no Streamlit
            if plan['needs_visualization']:
                st.info(f"🎯 Detectada solicitação de visualização: {plan['visualization_type'] or 'geral'}")
            
            results = []
            
            # Executar visualização se necessária
            if plan['needs_visualization'] and visualization_expert:
                st.info("🎨 Gerando visualizações...")
                
                try:
                    viz_type = plan['visualization_type']
                    
                    if viz_type == 'survival' or viz_type == 'survival_by_gender':
                        viz_result = visualization_expert.create_survival_chart_direct(data, user_question)
                    elif viz_type == 'correlation':
                        viz_result = visualization_expert.create_correlation_chart_direct(data)
                    elif viz_type == 'distribution':
                        column = plan['specific_columns'][0] if plan['specific_columns'] else None
                        viz_result = visualization_expert.create_distribution_chart_direct(data, column)
                    else:
                        # Visualização geral
                        viz_result = visualization_expert.create_general_visualization(data)
                    
                    results.append(viz_result)
                    
                except Exception as e:
                    error_msg = f"❌ Erro na visualização: {str(e)}"
                    st.error(error_msg)
                    results.append(error_msg)
            
            # Executar análise estatística se necessária
            if plan['needs_statistical_analysis'] and data_explorer_agent:
                if plan['response_strategy'] != 'visualization_focused':
                    st.info("📊 Executando análise complementar...")
                    
                    try:
                        # Análise básica quando não há agente específico disponível
                        analysis_result = self._basic_statistical_analysis(data, user_question, plan)
                        if analysis_result:
                            results.append(analysis_result)
                    except Exception as e:
                        st.warning(f"⚠️ Análise complementar falhou: {str(e)}")
            
            # Consolidar resultados
            if results:
                return self._consolidate_response(user_question, results, plan)
            else:
                return "❌ Não foi possível gerar resposta para sua pergunta. Tente reformular."
                
        except Exception as e:
            error_msg = f"Erro na coordenação: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def _basic_statistical_analysis(self, data: pd.DataFrame, question: str, plan: Dict[str, Any]) -> Optional[str]:
        """
        Análise estatística básica quando não há agente específico disponível
        """
        try:
            question_lower = question.lower()
            
            # Análise específica baseada na pergunta
            if any(word in question_lower for word in ['arquivo', 'dataset', 'qual', 'que dados']):
                return f"""
                **📋 Informações do Dataset:**
                • Dimensões: {data.shape[0]} linhas × {data.shape[1]} colunas
                • Colunas: {', '.join(data.columns.tolist())}
                • Tipos de dados: {len(data.select_dtypes(include=['number']).columns)} numéricas, {len(data.select_dtypes(include=['object']).columns)} categóricas
                """
            
            elif plan['visualization_type'] == 'survival_by_gender':
                if 'Sex' in data.columns and 'Survived' in data.columns:
                    # Já incluído na visualização, retornar informação complementar
                    total = len(data)
                    survivors = data['Survived'].sum()
                    return f"""
                    **📈 Contexto Adicional:**
                    • Dataset contém dados históricos do Titanic
                    • Total de registros analisados: {total}
                    • Taxa geral de sobrevivência: {survivors/total*100:.1f}%
                    """
            
            return None  # Sem análise adicional necessária
            
        except Exception as e:
            return f"Erro na análise básica: {str(e)}"
    
    def _consolidate_response(self, question: str, results: list, plan: Dict[str, Any]) -> str:
        """
        Consolida resultados em resposta coerente
        """
        response = f"## 🔍 Resposta para: '{question}'\n\n"
        
        # Adicionar contexto do plano se relevante
        if plan['visualization_type']:
            response += f"**🎯 Tipo de análise:** {plan['visualization_type']}\n\n"
        
        # Consolidar resultados
        for i, result in enumerate(results):
            response += result
            if i < len(results) - 1:
                response += "\n\n---\n\n"
        
        # Adicionar dicas baseadas no tipo de análise
        response += self._add_contextual_tips(plan)
        
        return response
    
    def _add_contextual_tips(self, plan: Dict[str, Any]) -> str:
        """
        Adiciona dicas contextuais baseadas no tipo de análise
        """
        tips = "\n\n**💡 Próximos passos sugeridos:**\n"
        
        if plan['visualization_type'] == 'survival_by_gender':
            tips += """
            • Investigue fatores adicionais: idade, classe socioeconômica
            • Analise padrões por localização no navio
            • Compare com outros desastres marítimos históricos
            """
        elif plan['visualization_type'] == 'correlation':
            tips += """
            • Explore as correlações mais fortes identificadas
            • Considere análises de regressão para relações causais
            • Investigue outliers que podem afetar correlações
            """
        elif plan['visualization_type'] == 'distribution':
            tips += """
            • Identifique e analise outliers na distribuição
            • Compare com distribuições teóricas conhecidas
            • Segmente análise por grupos categóricos
            """
        else:
            tips += """
            • Faça perguntas mais específicas sobre padrões identificados
            • Explore subgrupos dos dados para análises detalhadas
            • Solicite visualizações específicas conforme interesse
            """
        
        return tips
    
    def create_executive_summary(self, session_analyses: list) -> str:
        """
        Cria um resumo executivo de todas as análises da sessão
        """
        if not session_analyses:
            return "📝 Nenhuma análise realizada nesta sessão ainda."
        
        summary = "# 📋 Resumo Executivo da Sessão\n\n"
        summary += f"**Total de análises realizadas:** {len(session_analyses)}\n\n"
        
        # Categorizar análises
        viz_count = sum(1 for analysis in session_analyses if 'visualiz' in analysis.lower())
        stats_count = sum(1 for analysis in session_analyses if 'estatística' in analysis.lower())
        
        summary += f"• **Visualizações geradas:** {viz_count}\n"
        summary += f"• **Análises estatísticas:** {stats_count}\n\n"
        
        summary += "## 🔍 Principais Descobertas:\n\n"
        
        # Extrair insights principais (simplificado)
        key_findings = []
        for analysis in session_analyses[-3:]:  # Últimas 3 análises
            if 'sobrevivência' in analysis.lower():
                key_findings.append("• Padrões de sobrevivência identificados e analisados")
            elif 'correlação' in analysis.lower():
                key_findings.append("• Relacionamentos entre variáveis mapeados")
            elif 'distribuição' in analysis.lower():
                key_findings.append("• Características de distribuição dos dados exploradas")
        
        if key_findings:
            summary += "\n".join(set(key_findings))  # Remove duplicatas
        else:
            summary += "• Análise exploratória abrangente dos dados realizada"
        
        summary += "\n\n## 🎯 Recomendações:\n\n"
        summary += """
        • Continue explorando os padrões mais significativos encontrados
        • Considere análises mais específicas baseadas nos insights iniciais
        • Use as visualizações geradas para comunicar resultados
        • Documente as descobertas principais para referência futura
        """
        
        return summary
