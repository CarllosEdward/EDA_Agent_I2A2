import pandas as pd
import json

class VisualizationExpert:
    """
    Gera especificações de visualização em JSON a partir de um DataFrame.
    """
    
    def __init__(self, llm):
        """
        Inicializa o especialista em visualização.

        Args:
            llm: O modelo de linguagem a ser usado (atualmente não utilizado,
                 mas mantido para futuras integrações).
        """
        self.llm = llm

    def _create_plot_spec(self, plot_type: str, data: pd.DataFrame, params: dict) -> str:
        """
        Cria uma especificação de gráfico em formato JSON.
        """
        spec = {
            "plot_type": plot_type,
            "data": data.to_dict(orient='records'),
            "params": params
        }
        return json.dumps(spec)

    def create_survival_chart(self, data: pd.DataFrame, user_question: str = "") -> str:
        """
        Gera uma especificação para um gráfico de sobrevivência.
        """
        if 'Sex' in data.columns and 'Survived' in data.columns:
            # Especificação para um gráfico de barras de sobrevivência por gênero
            return self._create_plot_spec(
                plot_type='bar',
                data=data,
                params={
                    "x": "Sex",
                    "y": "Survived",
                    "color": "Survived",
                    "title": "Taxa de Sobrevivência por Gênero",
                    "barmode": "group"
                }
            )
        else:
            return self.create_general_visualization(data)

    def create_correlation_chart(self, data: pd.DataFrame) -> str:
        """
        Gera uma especificação para uma matriz de correlação (heatmap).
        """
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return json.dumps({"error": "Dataset possui menos de 2 colunas numéricas."})

        corr_matrix = data[numeric_cols].corr().to_dict()

        return self._create_plot_spec(
            plot_type='heatmap',
            data=pd.DataFrame(corr_matrix), # O heatmap espera um DataFrame
            params={
                "title": "Matriz de Correlação",
                "text_auto": True
            }
        )

    def create_distribution_chart(self, data: pd.DataFrame, column: str = None) -> str:
        """
        Gera uma especificação para um histograma de distribuição.
        """
        if column is None:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                column = numeric_cols[0]
            else:
                return json.dumps({"error": "Nenhuma coluna numérica encontrada."})

        if column not in data.columns:
            return json.dumps({"error": f"Coluna '{column}' não encontrada."})

        return self._create_plot_spec(
            plot_type='histogram',
            data=data[[column]],
            params={
                "x": column,
                "title": f"Distribuição de {column}",
                "marginal": "box"
            }
        )

    def create_general_visualization(self, data: pd.DataFrame) -> str:
        """
        Gera uma especificação para uma visualização geral (ex: pairplot).
        """
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return json.dumps({"error": "Visualização geral requer pelo menos 2 colunas numéricas."})

        return self._create_plot_spec(
            plot_type='pairplot',
            data=data,
            params={
                "title": "Visualização Geral das Relações entre Variáveis"
            }
        )
