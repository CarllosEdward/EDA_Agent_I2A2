from crewai import Task

def create_conclusion_task(agent, session_context: str = ""):
    """Cria tarefa de conclusões finais"""
    
    return Task(
        description=f"""
        Gerar conclusões consolidadas baseadas nas análises mais relevantes da sessão.
        
        Contexto da sessão: {session_context}
        
        Suas responsabilidades CONCISAS incluem:
        1. Identificar os TOP 3 insights mais importantes descobertos
        2. Destacar apenas anomalias ou padrões CRÍTICOS
        3. Resumir relações entre variáveis SIGNIFICATIVAS
        4. Fornecer 2-3 recomendações práticas baseadas nos dados
        
        IMPORTANTE: NÃO recupere TODAS as análises anteriores da memória. 
        Foque apenas nos insights essenciais e mais impactantes.
        
        FORMATO DE RESPOSTA (máximo 200 palavras):
        📊 DATASET: [resumo breve]
        💡 TOP 3 INSIGHTS:
        • [Insight mais importante]
        • [Segundo insight relevante]  
        • [Terceiro insight significativo]
        
        🎯 RECOMENDAÇÕES: [2-3 ações práticas]
        
        Seja direto, objetivo e focado no valor para o usuário.
        """,
        agent=agent,
        expected_output="Relatório executivo conciso (máximo 200 palavras) com os 3 principais insights e recomendações práticas",
        max_execution_time=60,  # Limite de 1 minuto
        output_format="markdown"
    )
