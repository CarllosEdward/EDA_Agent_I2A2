import pandas as pd
import json

class VisualizationExpert:
    """
    Generates visualization specifications in JSON format from a DataFrame.
    """
    
    def __init__(self, llm):
        """
        Initializes the visualization expert.

        Args:
            llm: The language model to be used (currently not utilized,
                 but kept for future integrations).
        """
        self.llm = llm

    def _create_plot_spec(self, plot_type: str, data: pd.DataFrame, params: dict) -> str:
        """
        Creates a chart specification in JSON format.
        """
        spec = {
            "plot_type": plot_type,
            "data": data.to_dict(orient='records'),
            "params": params
        }
        return json.dumps(spec)

    def create_survival_chart(self, data: pd.DataFrame, user_question: str = "") -> str:
        """
        Generates a specification for a survival chart.
        """
        if 'Sex' in data.columns and 'Survived' in data.columns:
            return self._create_plot_spec(
                plot_type='bar',
                data=data,
                params={
                    "x": "Sex",
                    "y": "Survived",
                    "color": "Survived",
                    "title": "The Fates of Souls by Gender",
                    "barmode": "group"
                }
            )
        else:
            return self.create_general_visualization(data)

    def create_correlation_chart(self, data: pd.DataFrame) -> str:
        """
        Generates a specification for a correlation matrix (heatmap).
        """
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return json.dumps({"error": "Dataset must have at least 2 numeric columns for correlation."})

        corr_matrix = data[numeric_cols].corr().to_dict()

        return self._create_plot_spec(
            plot_type='heatmap',
            data=pd.DataFrame(corr_matrix),
            params={
                "title": "Nexus of Correlations",
                "text_auto": True
            }
        )

    def create_distribution_chart(self, data: pd.DataFrame, column: str = None) -> str:
        """
        Generates a specification for a distribution histogram.
        """
        if column is None:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                column = numeric_cols[0]
            else:
                return json.dumps({"error": "No numeric columns found."})

        if column not in data.columns:
            return json.dumps({"error": f"Column '{column}' not found."})

        return self._create_plot_spec(
            plot_type='histogram',
            data=data[[column]],
            params={
                "x": column,
                "title": f"Distribution of {column}",
                "marginal": "box"
            }
        )

    def create_general_visualization(self, data: pd.DataFrame) -> str:
        """
        Generates a specification for a general visualization (e.g., pairplot).
        """
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return json.dumps({"error": "General visualization requires at least 2 numeric columns."})

        return self._create_plot_spec(
            plot_type='pairplot',
            data=data,
            params={
                "title": "Constellation of Variable Relationships"
            }
        )