from crewai import Task

def create_analysis_task(agent, question: str, context: str = ""):
    """Cria tarefa de análise específica"""
    
    return Task(
        description=f"""
        Realizar análise exploratória baseada na seguinte pergunta: "{question}"
        
        Contexto adicional: {context}
        
        Suas responsabilidades incluem:
        1. Interpretar a pergunta do usuário
        2. Determinar quais análises são necessárias
        3. Executar as análises apropriadas (estatísticas, correlações, outliers)
        4. Interpretar os resultados encontrados
        5. Identificar insights relevantes
        6. Documentar descobertas na memória
        
        Baseie suas análises nos dados previamente carregados e considere
        o histórico de análises anteriores disponível na memória.
        
        Forneça uma resposta clara e detalhada que responda diretamente
        à pergunta do usuário, incluindo números, estatísticas e insights.
        """,
        agent=agent,
        expected_output="Análise detalhada que responde à pergunta específica do usuário com dados concretos e insights"
    )
