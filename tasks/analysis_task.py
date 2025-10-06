from crewai import Task

def create_analysis_task(agent, question: str, context: str = ""):
    """Cria tarefa de análise específica com prompts melhorados."""
    
    return Task(
        description=f"""
        Realizar uma análise de dados profunda para responder a esta pergunta: "{question}".
        
        Sua principal tarefa é usar as ferramentas disponíveis para:
        
        - Analisar os dados e extrair informações relevantes.
        - Interpretar estatísticas, padrões e anomalias.
        - Sintetizar os resultados em um resumo conciso.
        
        Se o **contexto** fornecido for relevante, integre-o à sua análise: {context}.
        
        Seja direto e objetivo. Sua resposta final deve ser um insight claro e acionável,
        baseado nos dados. Não se limite a listar estatísticas; explique o que elas significam.
        """,
        agent=agent,
        expected_output="""
        Uma resposta clara e direta para a pergunta do usuário, contendo:
        
        - Os resultados numéricos ou estatísticos mais importantes.
        - Pelo menos 2-3 insights relevantes extraídos da análise.
        - Uma conclusão que sintetize as descobertas.
        
        A saída deve ser um texto formatado e fácil de ler.
        """
    )
