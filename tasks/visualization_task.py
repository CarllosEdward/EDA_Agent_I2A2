from crewai import Task

def create_visualization_task(agent, chart_request: str, data_context: str = ""):
    """Cria tarefa de visualização (evoluída)"""
    
    return Task(
        description=f"""
        Criar visualização para: {chart_request}
        
        Contexto dos dados: {data_context}
        
        Suas responsabilidades incluem:
        1. Determinar o tipo de gráfico mais adequado
        2. Selecionar as colunas relevantes para visualização
        3. Criar o gráfico com boa qualidade visual
        4. EXIBIR o gráfico diretamente na tela do Streamlit
        5. Disponibilizar download do gráfico gerado
        6. Fornecer interpretação da visualização
        
        IMPORTANTE: Use sempre os métodos do ChartGeneratorTool que automaticamente
        exibem os gráficos na tela e disponibilizam downloads. Não apenas mencione
        que "um gráfico foi gerado" - certifique-se que ele aparece visualmente.
        
        Considere as análises já realizadas e escolha visualizações que
        complementem e enriqueçam a compreensão dos dados.
        
        Para diferentes tipos de solicitação:
        - Sobrevivência + gênero: use create_survival_by_gender_chart()
        - Correlação: use create_correlation_heatmap()
        - Distribuição: use create_histogram()
        - Geral: escolha o mais apropriado baseado nos dados
        """,
        agent=agent,
        expected_output="""
        Resposta completa incluindo:
        1. Gráficos VISÍVEIS na tela do Streamlit
        2. Análise detalhada dos dados visualizados
        3. Interpretação dos padrões encontrados
        4. Botões de download disponíveis
        5. Insights acionáveis baseados na visualização
        
        FORMATO: Não apenas "gráfico gerado" mas sim gráficos realmente exibidos + análise textual
        """
    )

# NOVA FUNÇÃO: Criar task específica para Titanic
def create_titanic_survival_task(agent, user_question: str, data_context: str = ""):
    """Cria task específica para análise de sobrevivência do Titanic"""
    
    return Task(
        description=f"""
        Criar análise completa de sobrevivência do Titanic para: {user_question}
        
        Contexto: {data_context}
        
        EXECUTAR OBRIGATORIAMENTE:
        1. Verificar se colunas 'Sex' e 'Survived' existem no dataset
        2. Usar create_survival_by_gender_chart() para gerar visualização completa
        3. Exibir 4 gráficos diferentes:
           - Total de passageiros por gênero
           - Sobreviventes vs não sobreviventes
           - Taxa de sobrevivência percentual
           - Tabela resumo estatístico
        4. Fornecer análise detalhada dos padrões de sobrevivência
        5. Disponibilizar download da análise completa
        
        DADOS HISTÓRICOS: O dataset Titanic contém informações reais do naufrágio
        de 1912, incluindo dados demográficos e status de sobrevivência dos passageiros.
        
        INSIGHTS ESPERADOS:
        - Diferenças de sobrevivência entre homens e mulheres
        - Reflexo do protocolo "mulheres e crianças primeiro"
        - Análise quantitativa das taxas de sobrevivência
        """,
        agent=agent,
        expected_output="""
        Análise completa de sobrevivência do Titanic incluindo:
        
        1. **Gráficos visíveis na tela:**
           - 4 visualizações diferentes sobre sobrevivência por gênero
           - Tabela de estatísticas detalhadas
           
        2. **Análise quantitativa:**
           - Números absolutos de sobreviventes por gênero
           - Taxas percentuais de sobrevivência
           - Comparações estatísticas
           
        3. **Interpretação histórica:**
           - Contexto do protocolo marítimo da época
           - Insights sobre fatores sociais influenciadores
           
        4. **Elementos interativos:**
           - Botão de download da análise completa
           - Gráficos de alta qualidade para apresentação
        """
    )

# NOVA FUNÇÃO: Criar task de correlação
def create_correlation_analysis_task(agent, data_context: str = ""):
    """Cria task específica para análise de correlação"""
    
    return Task(
        description=f"""
        Criar análise completa de correlação entre variáveis do dataset.
        
        Contexto: {data_context}
        
        EXECUTAR:
        1. Verificar colunas numéricas disponíveis (mínimo 2)
        2. Usar create_correlation_heatmap() para gerar matriz visual
        3. Exibir heatmap interativo (Plotly) + versão estática (Matplotlib)
        4. Identificar correlações mais fortes (positivas e negativas)
        5. Interpretar significado das correlações encontradas
        6. Disponibilizar download do heatmap
        
        ANÁLISE INCLUI:
        - Matriz de correlação completa
        - Identificação de correlações > 0.7 ou < -0.7
        - Interpretação das relações entre variáveis
        - Sugestões para análises subsequentes
        """,
        agent=agent,
        expected_output="""
        Análise de correlação completa com:
        
        1. **Heatmap visível:** Matriz de correlação interativa na tela
        2. **Identificação:** Correlações mais fortes encontradas
        3. **Interpretação:** Significado das relações identificadas
        4. **Download:** Botão para salvar heatmap em alta qualidade
        5. **Próximos passos:** Sugestões baseadas nas correlações encontradas
        """
    )

# NOVA FUNÇÃO: Criar task de distribuição
def create_distribution_analysis_task(agent, column_name: str, data_context: str = ""):
    """Cria task específica para análise de distribuição"""
    
    return Task(
        description=f"""
        Criar análise de distribuição para a variável: {column_name}
        
        Contexto: {data_context}
        
        EXECUTAR:
        1. Verificar se coluna '{column_name}' existe e é numérica
        2. Usar create_histogram() para gerar visualização
        3. Calcular estatísticas descritivas completas
        4. Identificar possíveis outliers
        5. Analisar forma da distribuição (normal, assimétrica, etc.)
        6. Disponibilizar download do gráfico
        
        ANÁLISE INCLUI:
        - Histograma da distribuição
        - Média, mediana, desvio padrão
        - Quartis e valores extremos
        - Identificação de padrões na distribuição
        """,
        agent=agent,
        expected_output="""
        Análise de distribuição completa com:
        
        1. **Histograma visível:** Gráfico de distribuição na tela
        2. **Estatísticas:** Medidas de tendência central e dispersão
        3. **Outliers:** Identificação de valores atípicos se presentes
        4. **Interpretação:** Características da distribuição
        5. **Download:** Botão para salvar gráfico
        """
    )
