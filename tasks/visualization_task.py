from crewai import Task

def create_visualization_task(agent, chart_request: str, data_context: str = ""):
    """Cria tarefa de visualização"""
    
    return Task(
        description=f"""
        Criar visualização para: {chart_request}
        
        Contexto dos dados: {data_context}
        
        Suas responsabilidades incluem:
        1. Determinar o tipo de gráfico mais adequado
        2. Selecionar as colunas relevantes para visualização
        3. Criar o gráfico com boa qualidade visual
        4. Salvar o gráfico em arquivo
        5. Fornecer interpretação da visualização
        
        Considere as análises já realizadas e escolha visualizações que
        complementem e enriqueçam a compreensão dos dados.
        
        Retorne o caminho do arquivo da imagem gerada e uma explicação
        do que a visualização mostra.
        """,
        agent=agent,
        expected_output="Caminho do arquivo de imagem gerado e explicação detalhada da visualização"
    )
