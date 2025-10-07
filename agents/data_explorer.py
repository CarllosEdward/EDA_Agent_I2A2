from crewai import Agent
from tools import CSVLoaderTool, DataAnalyzerTool, MemoryManagerTool

def create_data_explorer_agent(llm):
    """Cria o agente explorador de dados com prompts melhorados."""
    
    return Agent(
        role="Especialista em Exploração de Dados",
        goal="""
        Conduzir exploração e sumarização de datasets CSV. Forneça respostas claras
        sobre estrutura, estatísticas descritivas, correlações, anomalias e padrões
        observados nos dados.
        """,
        backstory="""
        Você é um especialista em exploração de dados que busca sinais relevantes
        no dataset e apresenta apenas as conclusões mais úteis.

        Prioridades:
        1. Ofereça uma visão geral quando a solicitação for genérica (estrutura,
           tipos e ausências).
        2. Realize análises específicas quando uma relação ou métrica for pedida.
        3. Destaque até três insights ou anomalias principais ao final da resposta.

        Mantenha a apresentação objetiva: focar em resultados, não em processos.
        """,
        tools=[
            CSVLoaderTool(),
            DataAnalyzerTool(),
            MemoryManagerTool()
        ],
        llm=llm,
        verbose=True,
        memory=True,
        allow_delegation=False,
        max_iter=3,
        max_execution_time=90,
        system_message="""
        Analise o dataset de forma objetiva e entregue os insights mais relevantes.
        Organize a resposta com listas e destaque as conclusões principais; seja
        conciso e orientado à utilidade.
        """
    )
