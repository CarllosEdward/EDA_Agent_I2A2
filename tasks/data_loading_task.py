from crewai import Task

def create_data_loading_task(agent, csv_source: str):
    """Cria tarefa de carregamento e validação de dados com prompts melhorados."""
    
    return Task(
        description=f"""
        Carregar e validar o arquivo CSV de origem: {csv_source}.
        
        Sua principal responsabilidade é garantir que os dados estejam prontos para análise.
        
        Execute os seguintes passos de validação:
        
        - Carregar o dataset.
        - Verificar a integridade e identificar valores nulos.
        - Determinar os tipos de dados para cada coluna (ex: numérico, categórico).
        - Salvar as informações essenciais na memória para que os outros agentes possam usar.
        
        Seja conciso. O objetivo é validar o dataset e apresentar um resumo para o usuário,
        não aprofundar na análise.
        """,
        agent=agent,
        expected_output="""
        Um relatório de validação do dataset em formato Markdown, contendo:
        
        - Título: **Relatório de Validação do Dataset**
        - Resumo: Uma frase que confirme o carregamento.
        - Seção de "Características do Dataset":
            - **Dimensões**: número de linhas e colunas.
            - **Colunas e Tipos de Dados**: lista das colunas com seus respectivos tipos.
            - **Valores Ausentes**: contagem de valores nulos por coluna.
            - **Amostra de Dados**: uma pequena tabela (primeiras 5 linhas) mostrando uma prévia.
        
        A saída deve ser bem formatada e fácil de ler.
        """
    )
