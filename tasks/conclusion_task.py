from crewai import Task

def create_conclusion_task(agent, session_context: str = ""):
    """Cria tarefa de conclusões finais com prompts melhorados."""
    
    return Task(
        description=f"""
        Sintetize as análises da sessão para criar um resumo executivo.
        
        Sua missão é extrair os insights mais importantes e as recomendações
        mais valiosas, ignorando detalhes menores.
        
        Contexto da sessão: {session_context}
        
        Siga o formato abaixo, focando apenas no que é mais impactante:
        
        - Identifique os 3 **principais insights** descobertos.
        - Destaque **anomalias ou padrões críticos**.
        - Crie 2 a 3 **recomendações práticas** baseadas nas descobertas.
        
        Seja conciso e direto. A resposta deve ser um relatório de alto nível,
        sem a necessidade de recuperar toda a memória.
        """,
        agent=agent,
        expected_output="""
        Um relatório executivo conciso, com o seguinte formato:
        
        ## 📊 Resumo Executivo da Análise
        
        ### 💡 Principais Insights
        
        - [Insight mais importante da sessão]
        - [Segundo insight mais relevante]
        - [Terceiro insight significativo]
        
        ### 🎯 Recomendações
        
        - [Recomendação prática 1]
        - [Recomendação prática 2]
        """,
        max_execution_time=60,
        output_format="markdown"
    )
