# guide.py
# Um guia de deploy e otimizações em formato Python.
# Execute este arquivo para ver todas as instruções e soluções.

class DeployGuide:
    """
    Classe para organizar o guia de deploy no Railway e as otimizações do sistema EDA.
    """
    def __init__(self):
        self.project_status = "95% completo"
        self.platform = "Railway"
        self.repo_name = "eda-agente-inteligente"
        self.issues = {
            "memoria": "Agente tenta carregar toda a sessão anterior, consumindo memória e tokens.",
            "rate_limit_groq": "O modelo 'llama-3.1-8b-instant' atinge o limite de taxa de requisições por minuto.",
            "respostas_longas": "Respostas detalhadas consomem muitos tokens e tempo.",
            "modelos_invalidos": "Modelos Groq descontinuados em uso (ex: mixtral-8x7b-32768)."
        }

    def print_deploy_steps(self):
        """Imprime o passo a passo para o deploy no Railway."""
        print("---")
        print("### 🚀 Guia de Deploy Rápido no Railway")
        print(f"Status do Projeto: {self.project_status}, pronto para o deploy no {self.platform}.\n")
        
        print("1. Preparando o Repositório no GitHub")
        print("Garanta que seu repositório local e do GitHub estejam sincronizados.")
        print("```bash")
        print("git status")
        print("git add .")
        print(f"git commit -m \"🎯 Versão final: pronta para o deploy no {self.platform}\"")
        print("git push origin main")
        print("```\n")

        print("2. Ação de Deploy no Railway")
        print("Acesse https://railway.app, faça login e selecione:")
        print("New Project → Deploy from GitHub repo → Selecione seu repositório.")
        print(f"O Railway detectará a configuração de Python e Streamlit para {self.repo_name}.")
        
        print("\n3. Definindo as Variáveis de Ambiente")
        print("No painel do Railway, vá em 'Variables' e adicione:")
        print("```env")
        print("GROQ_API_KEY=gsk_sua_chave_groq")
        print("GROQ_MODEL=llama-3.1-8b-instant")
        print("OPENAI_API_KEY=sk_sua_chave_openai")
        print("OPENAI_MODEL=gpt-4o-mini")
        print("MAX_FILE_SIZE=50")
        print("TEMP_DIR=temp_files")
        print("PORT=8501")
        print("```\n")

        print("4. Ajustando o Comando de Inicialização")
        print("Em 'Settings → Deploy', defina o comando de inicialização:")
        print("```bash")
        print("streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0")
        print("```\n")

        print("5. Verificações Finais de Arquivos")
        print("Confirme a presença destes arquivos no seu repositório:")
        print("- ✅ railway.json")
        print("- ✅ Dockerfile")
        print("- ✅ requirements.txt")
        print("- ✅ streamlit_app.py")
        print("---")

    def print_issues_and_solutions(self):
        """Imprime os problemas identificados e as soluções propostas."""
        print("---")
        print("### 🔧 Problemas Identificados e Soluções Propostas")
        print("Seu sistema foi analisado e identificamos pontos de otimização:\n")
        
        for issue, description in self.issues.items():
            print(f"- **Problema:** {description}")
        
        print("\n### 🛠️ Estratégia de Otimização")
        print("A solução é fazer ajustes cirúrgicos para resolver os problemas de forma eficiente.\n")

        print("1. **Modelos Groq Atualizados**")
        print("Modelos como 'llama-3.1-70b-versatile' foram descontinuados. A lista de modelos válidos para 2025 é 'llama-3.1-8b-instant', entre outros.")
        
        print("\n2. **Melhorias de Desempenho e Confiabilidade**")
        print("As seguintes otimizações foram ou serão implementadas:")
        print("- **Memória:** Limite de 10 análises recentes, com truncamento de texto.")
        print("- **Tokens:** Respostas mais concisas e padrão de tokens mais conservador.")
        print("- **Rate Limit Handler:** Sistema inteligente de retentativa e fallback para OpenAI em caso de falha do Groq.")
        print("- **Formatos:** Melhoria na formatação de saída para facilitar a leitura (ex: primeiras 5 linhas do CSV).")
        print("---")
        
    def get_final_files_list(self):
        """Retorna uma lista dos arquivos a serem corrigidos."""
        print("### 📝 Arquivos a Serem Sincronizados com as Correções\n")
        files = [
            "agents/coordenador.py",
            "agents/data_explorer.py",
            "agents/visualization_expert.py",
            "tasks/data_loading_task.py",
            "tasks/analysis_task.py",
            "tasks/visualization_task.py",
            "tasks/conclusion_task.py",
            "tools/memory_manager.py",
            "utils/config.py",
            "streamlit_app.py",
            "main.py",
            "utils/rate_limit_handler.py"
        ]
        for file in files:
            print(f"- ✅ {file}")

# ---
# Ponto de entrada do script
# ---
if __name__ == "__main__":
    guide = DeployGuide()
    guide.print_deploy_steps()
    guide.print_issues_and_solutions()
    guide.get_final_files_list()
