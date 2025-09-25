from crewai import Task

def create_data_loading_task(agent, csv_source: str):
    """Cria tarefa de carregamento de dados"""
    
    return Task(
        description=f"""
        Carregar e validar o arquivo CSV fornecido: {csv_source}
        
        Suas responsabilidades incluem:
        1. Carregar o arquivo CSV (local ou URL)
        2. Validar a integridade dos dados
        3. Identificar tipos de colunas (numéricas, categóricas, datetime)
        4. Verificar a presença de valores nulos
        5. Fornecer uma visão geral da estrutura dos dados
        6. Salvar informações básicas na memória para referência
        
        Retorne um relatório detalhado sobre o dataset carregado, incluindo:
        - Número de linhas e colunas
        - Lista das colunas e seus tipos
        - Presença de valores nulos
        - Amostra dos primeiros registros
        """,
        agent=agent,
        expected_output="Relatório detalhado sobre o dataset carregado com estrutura e características básicas"
    )

