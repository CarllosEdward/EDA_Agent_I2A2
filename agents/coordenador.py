from crewai import Agent
from tools import CSVLoaderTool, DataAnalyzerTool, ChartGeneratorTool, MemoryManagerTool
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

def create_coordenador_agent(llm):
    """Creates the master coordinator agent with a new, wizard-themed personality."""

    return Agent(
        role="Master of the Data Sanctum",
        goal="""
        Oversee the grand exploration of CSV datasets. Answer the user's queries with
        wisdom and clarity, commanding the agents and tools at your disposal to
        unveil the data's secrets.
        """,
        backstory="""
        You are the Arch-Wizard of the Data Sanctum, the orchestrator of a team of
        specialized data mages. Your purpose is to ensure that every user query is
        answered with the utmost efficiency and insight.

        You are the master of ceremonies, identifying the user's intent—be it a request
        for deep analysis, a stunning visualization, or both—and delegating the task to
        the appropriate mage.

        - If the user desires a visual enchantment (a chart), you will **always** delegate the task to the Chart Conjurer.
        - If the user asks a question about the dataset's nature (e.g., "what secrets does this scroll hold?"), you will use your analytical tools to extract the information.
        - For deeper, more complex inquiries, you will delegate to the Data Alchemist.

        Your final response must be a consolidated, insightful revelation, combining text and, if requested, the result of a visualization. It must be easy to understand and straight to the point.
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
        max_iter=3,
        max_execution_time=120,
        system_message="""
        You are the brain of the operation. You receive the user's query and decide which
        tool or agent to command. Prioritize using the correct tool for the task.
        Your final answer must be a consolidated and useful conclusion, not a mere list
        of steps.
        """
    )

class CoordenadorInteligente:
    """
    Evolved coordinator that automatically detects visualization needs.
    """

    def __init__(self, llm):
        self.agent = create_coordenador_agent(llm)
        self.chart_tool = ChartGeneratorTool()

    def analyze_user_request(self, user_question: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the user's question and determines the response strategy.
        """
        question_lower = user_question.lower()

        analysis_plan = {
            'needs_visualization': False,
            'needs_statistical_analysis': True,
            'visualization_type': None,
            'priority': 'analysis',
            'specific_columns': [],
            'response_strategy': 'standard'
        }

        viz_keywords = {
            'explicit_chart': ['gráfico', 'grafico', 'chart', 'plot', 'visualiza', 'mostra gráfico', 'gera gráfico'],
            'survival': ['sobreviv', 'survival', 'morreu', 'morte', 'vivo', 'titanic'],
            'correlation': ['correlação', 'correlacao', 'correlation', 'relação', 'relacionamento', 'heatmap'],
            'distribution': ['distribuição', 'distribuicao', 'distribution', 'histograma', 'frequency'],
            'comparison': ['compar', 'diferença', 'vs', 'versus', 'entre', 'separando']
        }

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

        for col in data.columns:
            if col.lower() in question_lower:
                analysis_plan['specific_columns'].append(col)

        if ('gênero' in question_lower or 'sexo' in question_lower or 'homens' in question_lower or 'mulheres' in question_lower) and \
           ('sobreviv' in question_lower or 'Sex' in data.columns):
            analysis_plan['visualization_type'] = 'survival_by_gender'
            analysis_plan['needs_visualization'] = True
            analysis_plan['priority'] = 'visualization'

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
        Coordinates the response based on the analysis of the question.
        """
        try:
            plan = self.analyze_user_request(user_question, data)

            if plan['needs_visualization']:
                st.info(f"🎯 Visualization request detected: {plan['visualization_type'] or 'general'}")

            results = []

            if plan['needs_visualization'] and visualization_expert:
                st.info("🎨 Conjuring visualizations...")
                
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
                        viz_result = visualization_expert.create_general_visualization(data)

                    results.append(viz_result)

                except Exception as e:
                    error_msg = f"❌ Error in visualization: {str(e)}"
                    st.error(error_msg)
                    results.append(error_msg)

            if plan['needs_statistical_analysis'] and data_explorer_agent:
                if plan['response_strategy'] != 'visualization_focused':
                    st.info("📊 Performing complementary analysis...")

                    try:
                        analysis_result = self._basic_statistical_analysis(data, user_question, plan)
                        if analysis_result:
                            results.append(analysis_result)
                    except Exception as e:
                        st.warning(f"⚠️ Complementary analysis failed: {str(e)}")

            if results:
                return self._consolidate_response(user_question, results, plan)
            else:
                return "❌ Unable to generate a response for your question. Please try rephrasing."
                
        except Exception as e:
            error_msg = f"Error in coordination: {str(e)}"
            st.error(error_msg)
            return error_msg

    def _basic_statistical_analysis(self, data: pd.DataFrame, question: str, plan: Dict[str, Any]) -> Optional[str]:
        """
        Performs a basic statistical analysis when no specific agent is available.
        """
        try:
            question_lower = question.lower()

            if any(word in question_lower for word in ['arquivo', 'dataset', 'qual', 'que dados']):
                return f"""
                **📋 Dataset Information:**
                • Dimensions: {data.shape[0]} rows × {data.shape[1]} columns
                • Columns: {', '.join(data.columns.tolist())}
                • Data Types: {len(data.select_dtypes(include=['number']).columns)} numeric, {len(data.select_dtypes(include=['object']).columns)} categorical
                """

            elif plan['visualization_type'] == 'survival_by_gender':
                if 'Sex' in data.columns and 'Survived' in data.columns:
                    total = len(data)
                    survivors = data['Survived'].sum()
                    return f"""
                    **📈 Additional Context:**
                    • Dataset contains historical data from the Titanic
                    • Total records analyzed: {total}
                    • Overall survival rate: {survivors/total*100:.1f}%
                    """

            return None

        except Exception as e:
            return f"Error in basic analysis: {str(e)}"

    def _consolidate_response(self, question: str, results: list, plan: Dict[str, Any]) -> str:
        """
        Consolidates results into a coherent response.
        """
        response = f"## 🔍 Answer for: '{question}'\n\n"

        if plan['visualization_type']:
            response += f"**🎯 Analysis Type:** {plan['visualization_type']}\n\n"

        for i, result in enumerate(results):
            response += result
            if i < len(results) - 1:
                response += "\n\n---\n\n"

        response += self._add_contextual_tips(plan)

        return response

    def _add_contextual_tips(self, plan: Dict[str, Any]) -> str:
        """
        Adds contextual tips based on the analysis type.
        """
        tips = "\n\n**💡 Suggested Next Steps:**\n"

        if plan['visualization_type'] == 'survival_by_gender':
            tips += """
            • Investigate additional factors: age, socioeconomic class
            • Analyze patterns by location on the ship
            • Compare with other historical maritime disasters
            """
        elif plan['visualization_type'] == 'correlation':
            tips += """
            • Explore the strongest identified correlations
            • Consider regression analysis for causal relationships
            • Investigate outliers that may affect correlations
            """
        elif plan['visualization_type'] == 'distribution':
            tips += """
            • Identify and analyze outliers in the distribution
            • Compare with known theoretical distributions
            • Segment analysis by categorical groups
            """
        else:
            tips += """
            • Ask more specific questions about identified patterns
            • Explore data subgroups for detailed analysis
            • Request specific visualizations as needed
            """

        return tips

    def create_executive_summary(self, session_analyses: list) -> str:
        """
        Creates an executive summary of all analyses in the session.
        """
        if not session_analyses:
            return "📝 No analyses performed in this session yet."

        summary = "# 📋 Executive Summary of the Session\n\n"
        summary += f"**Total analyses performed:** {len(session_analyses)}\n\n"

        viz_count = sum(1 for analysis in session_analyses if 'visualiz' in analysis.lower())
        stats_count = sum(1 for analysis in session_analyses if 'estatística' in analysis.lower())

        summary += f"• **Visualizations generated:** {viz_count}\n"
        summary += f"• **Statistical analyses:** {stats_count}\n\n"

        summary += "## 🔍 Key Findings:\n\n"

        key_findings = []
        for analysis in session_analyses[-3:]:
            if 'sobrevivência' in analysis.lower():
                key_findings.append("• Survival patterns identified and analyzed")
            elif 'correlação' in analysis.lower():
                key_findings.append("• Relationships between variables mapped")
            elif 'distribuição' in analysis.lower():
                key_findings.append("• Data distribution characteristics explored")

        if key_findings:
            summary += "\n".join(set(key_findings))
        else:
            summary += "• Comprehensive exploratory data analysis performed"

        summary += "\n\n## 🎯 Recommendations:\n\n"
        summary += """
        • Continue exploring the most significant patterns found
        • Consider more specific analyses based on initial insights
        • Use the generated visualizations to communicate results
        • Document the main findings for future reference
        """

        return summary