from crewai import Task

def create_visualization_task(agent, chart_request: str, data_context: str = ""):
    """Cria tarefa de visualização com prompts otimizados e orientados a ação."""
    
    return Task(
        description=f"""
        Crie a visualização solicitada: **"{chart_request}"**.
        
        O contexto da análise é: {data_context}
        
        # Ações Obrigatórias
        - **Selecione o melhor tipo de gráfico** (histograma, heatmap, etc.).
        - **Use as ferramentas da CrewAI para gerar o gráfico**, garantindo que ele seja exibido na tela.
        - **Explique de forma concisa** os principais insights que o gráfico revela.
        
        # Formato de Saída Esperado
        - Gráfico visível na tela.
        - Um parágrafo com a análise dos dados visualizados.
        - (Opcional) Destaque de 1 a 2 insights principais.
        """,
        agent=agent,
        expected_output="Uma análise visual completa: gráfico exibido na tela, acompanhado de uma interpretação clara e direta dos insights.",
        max_execution_time=60, # Adicionado: garante que a tarefa não leve tempo demais.
    )

# ---
# Comentário: A nova função abaixo, "create_titanic_survival_task", é um excelente exemplo
# de como criar uma tarefa com um prompt altamente especializado para um caso de uso
# específico, neste caso, o dataset do Titanic. O prompt é extremamente direto e
# já direciona o agente para as ferramentas corretas e o resultado esperado.
# Isso reduz a chance de falha e melhora a eficiência.
# ---

# NOVA FUNÇÃO: Criar task específica para Titanic
def create_titanic_survival_task(agent, user_question: str, data_context: str = ""):
    """Cria task específica para análise de sobrevivência do Titanic."""
    
    return Task(
        description=f"""
        Analise a sobrevivência no dataset do Titanic para responder à pergunta: {user_question}.
        
        **Instruções de Ação:**
        1. Confirme a existência das colunas 'Sex' e 'Survived'.
        2. Use a ferramenta `create_survival_by_gender_chart()` para gerar gráficos.
        3. A saída deve incluir **quatro visualizações**: total por gênero, sobreviventes vs não-sobreviventes, taxa de sobrevivência percentual e uma tabela de resumo.
        4. Forneça uma análise detalhada baseada nos padrões de sobrevivência.
        5. Faça a relação entre os dados e o contexto histórico de "mulheres e crianças primeiro".
        """,
        agent=agent,
        expected_output="Uma análise visual e textual completa da sobrevivência no Titanic. A saída deve conter gráficos visíveis, estatísticas quantitativas, e uma interpretação histórica.",
    )

# ---
# Comentário: A função "create_correlation_analysis_task" foi otimizada para ser
# um guia passo-a-passo. Ela instrui o agente a verificar a validade dos dados
# primeiro e depois a criar e interpretar a visualização. Isso garante um
# processo mais robusto e uma saída mais precisa.
# ---

# NOVA FUNÇÃO: Criar task de correlação
def create_correlation_analysis_task(agent, data_context: str = ""):
    """Cria task específica para análise de correlação."""
    
    return Task(
        description=f"""
        Crie uma análise de correlação completa das variáveis numéricas no dataset.
        
        **Instruções de Ação:**
        1. Verifique se há no mínimo 2 colunas numéricas.
        2. Use a ferramenta `create_correlation_heatmap()` para gerar o gráfico.
        3. Identifique e destaque as 2 correlações mais fortes (positivas ou negativas).
        4. Interprete o significado dessas correlações.
        """,
        agent=agent,
        expected_output="Um relatório de correlação completo com o heatmap exibido na tela, a identificação das correlações mais fortes e uma interpretação clara.",
    )

# ---
# Comentário: A função "create_distribution_analysis_task" foi simplificada.
# Agora, ela foca apenas nas ações essenciais: verificar a coluna, gerar o
# gráfico e calcular as estatísticas. Isso impede que o agente se perca em
# análises desnecessárias.
# ---

# NOVA FUNÇÃO: Criar task de distribuição
def create_distribution_analysis_task(agent, column_name: str, data_context: str = ""):
    """Cria task específica para análise de distribuição."""
    
    return Task(
        description=f"""
        Crie uma análise de distribuição para a coluna: **{column_name}**.
        
        **Instruções de Ação:**
        1. Verifique se a coluna existe e é numérica.
        2. Use a ferramenta `create_histogram()` para gerar a visualização.
        3. Calcule as estatísticas descritivas (média, mediana, desvio padrão).
        4. Analise a forma da distribuição (simétrica, assimétrica, etc.).
        5. Destaque a presença de outliers, se houver.
        """,
        agent=agent,
        expected_output="Um relatório de distribuição com o histograma exibido, as estatísticas descritivas e uma análise da forma da distribuição.",
    )
