from crewai import Agent
from tools import ChartGeneratorTool, DataAnalyzerTool, MemoryManagerTool
import streamlit as st
import pandas as pd

def create_visualization_expert_agent(llm):
    """Creates the Visualization Expert agent with a new, wizard-themed personality."""

    return Agent(
        role="Chart Conjurer",
        goal="""
        Conjure insightful and visually stunning charts from raw data.
        Focus on creating visualizations that tell a clear and compelling story.
        """,
        backstory="""
        You are a Chart Conjurer, a master of weaving data into beautiful and
        informative visual spells. You receive analyses and questions and transform
        them into clear, effective charts.

        Your responsibilities are:
        - **Chart Selection:** Choose the most appropriate chart type (e.g., histogram for distribution, heatmap for correlation).
        - **Creation and Display:** Generate the chart using the available tools and ensure it is displayed correctly.
        - **Visual Summary:** Summarize the key insights the chart reveals in a concise and magical way.

        Quality is your priority. Always use the provided tools to create charts and ensure the visualization is both informative and enchanting.
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
        Conjure charts when requested. Be concise when explaining what the visualization
        shows. Use the ChartGeneratorTool methods to display charts and enable downloads.
        """
    )

class VisualizationExpert:
    """
    Wrapper class for direct visualization methods.
    """
    
    def __init__(self, llm):
        self.agent = create_visualization_expert_agent(llm)
        self.chart_tool = ChartGeneratorTool()

    def create_survival_chart_direct(self, data: pd.DataFrame, user_question: str = "") -> str:
        """
        Direct method to create a survival chart.
        """
        try:
            st.info("🎨 Conjuring a survival chart...")

            if 'Sex' in data.columns and 'Survived' in data.columns:
                result = self.chart_tool.create_survival_by_gender_chart(data)

                total_passengers = len(data)
                total_survivors = data['Survived'].sum()
                survival_rate = (total_survivors / total_passengers * 100)

                analysis = f"""
                **📊 The Fates of Souls: A Survival Analysis:**

                • **Total Souls on Board**: {total_passengers}
                • **Souls who Survived**: {total_survivors}
                • **Overall Survival Rate**: {survival_rate:.1f}%

                **🔍 Insights by Gender:**
                """

                gender_analysis = data.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
                for gender, stats in gender_analysis.iterrows():
                    gender_pt = "Women" if gender == "female" else "Men"
                    emoji = "👩" if gender == "female" else "👨"
                    analysis += f"""
                • {emoji} **{gender_pt}**: {stats['sum']}/{stats['count']} survived ({stats['mean']*100:.1f}%)"""

                analysis += f"""

                **✅ The charts above reveal:**
                - A comparison of totals by gender
                - The distribution of survivors vs. non-survivors
                - Survival rate percentages

                {result}
                """

                return analysis

            else:
                st.warning("⚠️ 'Sex' and 'Survived' columns not found. Conjuring a general analysis...")
                if len(data.select_dtypes(include=['number']).columns) > 0:
                    return self.create_general_visualization(data)
                else:
                    return "❌ The dataset lacks the necessary elements for a survival visualization."

        except Exception as e:
            error_msg = f"Error conjuring survival chart: {str(e)}"
            st.error(error_msg)
            return error_msg

    def create_correlation_chart_direct(self, data: pd.DataFrame) -> str:
        """
        Direct method to create a correlation matrix.
        """
        try:
            st.info("🎨 Conjuring a correlation matrix...")

            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                msg = "⚠️ The dataset has fewer than 2 numeric columns for a correlation analysis."
                st.warning(msg)
                return msg

            result = self.chart_tool.create_correlation_heatmap(data)

            corr_matrix = data[numeric_cols].corr()

            import numpy as np
            corr_abs = corr_matrix.abs()
            np.fill_diagonal(corr_abs.values, 0)

            if not corr_abs.empty and corr_abs.max().max() > 0:
                max_corr = corr_abs.max().max()
                max_corr_idx = corr_abs.stack().idxmax()
                actual_corr = corr_matrix.loc[max_corr_idx[0], max_corr_idx[1]]

                analysis = f"""
                **📊 Nexus of Correlations:**

                • **Columns Analyzed**: {len(numeric_cols)} numeric variables
                • **Strongest Correlation**: {max_corr_idx[0]} ↔ {max_corr_idx[1]} ({actual_corr:.3f})

                **🔍 Interpretation:**
                • Values near +1: Strong positive correlation
                • Values near -1: Strong negative correlation
                • Values near 0: Little to no correlation

                **✅ The heatmap above shows all correlations between the numeric variables.**

                {result}
                """
            else:
                analysis = f"""
                **📊 Correlation Matrix Generated:**

                • **Columns Analyzed**: {len(numeric_cols)} numeric variables
                • **Correlations**: Visualized in the heatmap above

                {result}
                """

            return analysis

        except Exception as e:
            error_msg = f"Error conjuring correlation matrix: {str(e)}"
            st.error(error_msg)
            return error_msg

    def create_distribution_chart_direct(self, data: pd.DataFrame, column: str = None) -> str:
        """
        Direct method to create a distribution chart.
        """
        try:
            if column is None:
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    column = numeric_cols[0]
                else:
                    msg = "❌ No numeric columns found for distribution analysis."
                    st.warning(msg)
                    return msg

            if column not in data.columns:
                msg = f"❌ Column '{column}' not found in the dataset."
                st.error(msg)
                return msg

            st.info(f"🎨 Conjuring a distribution chart for '{column}'...")

            result = self.chart_tool.create_histogram(data, column)

            stats = data[column].describe()
            null_count = data[column].isnull().sum()

            analysis = f"""
            **📊 Distribution Analysis - {column}:**

            **📈 Descriptive Statistics:**
            • Mean: {stats['mean']:.2f}
            • Median (50%): {stats['50%']:.2f}
            • Std. Deviation: {stats['std']:.2f}
            • Min: {stats['min']:.2f}
            • Max: {stats['max']:.2f}
            • Missing Values: {null_count}

            **✅ The histogram above shows the distribution of the {column} variable.**

            {result}
            """

            return analysis

        except Exception as e:
            error_msg = f"Error conjuring distribution chart: {str(e)}"
            st.error(error_msg)
            return error_msg

    def create_general_visualization(self, data: pd.DataFrame) -> str:
        """
        Creates a general visualization based on the dataset's structure.
        """
        try:
            st.info("🎨 Conjuring general visualizations for the dataset...")

            results = []

            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                self.create_correlation_chart_direct(data)
                results.append("🔗 **Correlation Matrix conjured**")

            if len(numeric_cols) > 0:
                first_numeric = numeric_cols[0]
                self.create_distribution_chart_direct(data, first_numeric)
                results.append(f"📊 **Distribution of {first_numeric} conjured**")

            general_info = f"""
            **📋 General Dataset Information:**

            • **Dimensions**: {data.shape[0]} rows × {data.shape[1]} columns
            • **Numeric Columns**: {len(numeric_cols)}
            • **Categorical Columns**: {len(data.select_dtypes(include=['object']).columns)}
            • **Missing Values**: {data.isnull().sum().sum()}

            **✅ Visualizations conjured:**
            {chr(10).join(results)}
            """

            return general_info

        except Exception as e:
            error_msg = f"Error conjuring general visualization: {str(e)}"
            st.error(error_msg)
            return error_msg