from crewai import Agent
import pandas as pd
import json

class CoordenadorInteligente:
    """
    A smart coordinator that determines whether a user's request
    requires a data analysis response or a visualization.
    """

    def __init__(self, llm):
        """
        Initializes the coordinator.
        
        Args:
            llm: The language model to be used for decision making.
        """
        self.llm = llm

    def _is_visualization_request(self, user_question: str) -> bool:
        """
        Determines if the user's question is asking for a visualization.
        This is a simplified check based on keywords. A more advanced
        implementation could use an LLM call for more nuanced detection.
        """
        viz_keywords = [
            'gráfico', 'grafico', 'plot', 'chart', 'visualiza',
            'correlação', 'correlacao', 'distribuição', 'distribuicao',
            'heatmap', 'histograma', 'mostra', 'gera', 'cria', 'ilustra',
            'matriz', 'scatter', 'compar'
        ]
        return any(keyword in user_question.lower() for keyword in viz_keywords)

    def coordinate_response(self, user_question: str, data: pd.DataFrame,
                          data_explorer_agent, visualization_expert) -> str:
        """
        Coordinates the response by routing the request to the appropriate expert.

        Args:
            user_question: The question asked by the user.
            data: The DataFrame containing the data to be analyzed.
            data_explorer_agent: The agent responsible for text-based data analysis.
            visualization_expert: The expert responsible for generating plot specifications.
            
        Returns:
            A string containing either a text-based analysis or a JSON plot specification.
        """
        try:
            if self._is_visualization_request(user_question):
                # The user wants a plot. Delegate to the visualization expert.
                # The expert will decide the best plot type.
                # For simplicity, we'll call a general method. A more complex
                # system could route to more specific methods based on the question.
                
                # A simple routing logic can be added here if needed.
                if 'correlação' in user_question.lower() or 'correlation' in user_question.lower():
                    return visualization_expert.create_correlation_chart(data)
                elif 'distribuição' in user_question.lower() or 'distribution' in user_question.lower():
                    # A more robust implementation would extract the column name.
                    return visualization_expert.create_distribution_chart(data)
                elif 'sobreviv' in user_question.lower():
                     return visualization_expert.create_survival_chart(data, user_question)
                else:
                    # Default to a general visualization if specific keywords aren't found.
                    return visualization_expert.create_general_visualization(data)

            else:
                # The user wants a text-based analysis. Delegate to the data explorer.
                # We need to create a simple task for the data explorer agent.
                from crewai import Task, Crew, Process

                analysis_task = Task(
                    description=f"Analyze the dataset to answer the following question: '{user_question}'. The data has columns: {', '.join(data.columns)}.",
                    agent=data_explorer_agent,
                    expected_output="A concise, text-based answer to the user's question."
                )

                crew = Crew(
                    agents=[data_explorer_agent],
                    tasks=[analysis_task],
                    process=Process.sequential,
                    verbose=False
                )
                
                result = crew.kickoff()
                return result

        except Exception as e:
            # If anything goes wrong, return a JSON error message.
            # The UI can then display this to the user.
            return json.dumps({"error": f"An error occurred in the coordinator: {str(e)}"})