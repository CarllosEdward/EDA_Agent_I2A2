import os
import pandas as pd
from crewai import Crew, Process
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from utils.config import Config
from agents.coordenador import create_coordenador_agent, CoordenadorInteligente
from agents.data_explorer import create_data_explorer_agent
from agents.visualization_expert import create_visualization_expert_agent, VisualizationExpert
from tasks import create_data_loading_task, create_analysis_task, create_visualization_task, create_conclusion_task
from tasks.visualization_task import create_titanic_survival_task, create_correlation_analysis_task  # ADICIONADO
from utils.helpers import ensure_directories
from datetime import datetime
import streamlit as st  # ADICIONADO: Para feedback visual

class EDACrewSystem:
    """Sistema principal de análise EDA com CrewAI - Versão Final com Visualizações Integradas"""
    
    def __init__(self, llm_provider: str = "openai", model_name: str = None, max_tokens: int = 1500):
        """
        Inicializa o sistema EDA (mantido + visualizações)
        Args:
            llm_provider: 'groq' ou 'openai' - PADRÃO MANTIDO PARA OPENAI
            model_name: nome específico do modelo
            max_tokens: limite de tokens para respostas
        """
        print(f"🔧 Inicializando EDACrewSystem: {llm_provider} - {model_name} (max_tokens: {max_tokens})")
        
        ensure_directories()
        self.llm_provider = llm_provider
        self.model_name = model_name or self._get_default_model(llm_provider)
        self.max_tokens = max_tokens
        self.llm = self._setup_llm(llm_provider, self.model_name, max_tokens)
        
        # Contexto do dataset atual (mantido)
        self.current_dataset = None
        self.dataset_info = {
            'name': '',
            'source': '',
            'shape': None,
            'columns': [],
            'dtypes': {},
            'loaded_at': None,
            'sample_data': []
        }
        
        print(f"✅ Sistema configurado: {self.llm_provider} - {self.model_name}")
        
        # Inicializa agentes (mantido)
        self.coordenador = create_coordenador_agent(self.llm)
        self.data_explorer = create_data_explorer_agent(self.llm)
        self.visualization_expert = create_visualization_expert_agent(self.llm)

        # NOVO: Inicializar componentes evoluídos
        self.coordenador_inteligente = CoordenadorInteligente(self.llm)
        self.visualization_expert_direct = VisualizationExpert(self.llm)
    
    def _get_default_model(self, provider: str) -> str:
        """Retorna modelo padrão para o provider (mantido)"""
        defaults = {
            "groq": "llama-3.1-8b-instant",  # MANTIDO
            "openai": "gpt-3.5-turbo"        # MANTIDO
        }
        return defaults.get(provider.lower(), "gpt-3.5-turbo")
    
    def _setup_llm(self, provider: str, model_name: str, max_tokens: int):
        """Configura o modelo LLM (mantido)"""
        print(f"🔧 Configurando LLM: {provider} - {model_name} (tokens: {max_tokens})")
        
        if provider.lower() == "groq":
            api_key = Config.GROQ_API_KEY
            if not api_key:
                raise ValueError("GROQ_API_KEY não configurada")
            
            # MANTIDO: Verificar se modelo precisa de prefixo automático
            if not any(model_name.startswith(prefix) for prefix in ["groq/", "openai/", "qwen/", "moonshotai/"]):
                groq_model = f"groq/{model_name}"
            else:
                groq_model = model_name
                
            print(f"🔧 Modelo Groq configurado: {groq_model}")
            
            # MANTIDO: Limites de tokens mais conservadores para Groq
            safe_max_tokens = min(max_tokens, 500)  # Forçar limite baixo para evitar rate limit
            if safe_max_tokens != max_tokens:
                print(f"⚠️ Tokens reduzidos para Groq: {max_tokens} → {safe_max_tokens}")
            
            return ChatGroq(
                api_key=api_key,
                model=groq_model,  # Usar modelo com prefixo correto
                temperature=0.1,
                max_tokens=safe_max_tokens,  # Usar limite seguro
                timeout=30,        # MANTIDO: de 60 para 30
                max_retries=1,     # MANTIDO: de 2 para 1
                streaming=False
            )
        
        elif provider.lower() == "openai":
            api_key = Config.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY não configurada")
            
            print(f"🔧 Modelo OpenAI configurado: {model_name}")
            
            return ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0.1,
                max_tokens=max_tokens,  # Manter limite original para OpenAI
                timeout=60,
                max_retries=2
            )
        
        else:
            raise ValueError("Provider deve ser 'groq' ou 'openai'")
    
    def load_dataset(self, csv_source: str) -> str:
        """Carrega dataset CSV com contexto completo (mantido + melhorado)"""
        try:
            print(f"📄 Carregando: {csv_source} com {self.llm_provider}-{self.model_name}")
            
            # MANTIDO: Verificação preventiva para Groq
            if self.llm_provider == "groq" and self.max_tokens > 400:
                print("⚠️ Groq com tokens altos - reduzindo para evitar rate limit")
                old_tokens = self.max_tokens
                self.max_tokens = min(self.max_tokens, 400)
                # Reconfigurar LLM com limite menor
                self.llm = self._setup_llm(self.llm_provider, self.model_name, self.max_tokens)
                # NOVO: Recriar também componentes evoluídos
                self.coordenador_inteligente = CoordenadorInteligente(self.llm)
                self.visualization_expert_direct = VisualizationExpert(self.llm)
                print(f"🔧 Tokens otimizados: {old_tokens} → {self.max_tokens}")
            
            # MANTIDO: Carregar dataset internamente
            try:
                if csv_source.startswith(('http://', 'https://')):
                    from utils.helpers import download_csv_from_url
                    temp_path = download_csv_from_url(csv_source)
                    self.current_dataset = pd.read_csv(temp_path)
                    os.unlink(temp_path)
                else:
                    self.current_dataset = pd.read_csv(csv_source)
                
                # MANTIDO: Extrair nome do arquivo
                if csv_source.startswith(('http://', 'https://')):
                    dataset_name = csv_source.split('/')[-1]
                    if '?' in dataset_name:
                        dataset_name = dataset_name.split('?')[0]
                else:
                    dataset_name = os.path.basename(csv_source)
                
                if not dataset_name.endswith('.csv'):
                    if '.' not in dataset_name:
                        dataset_name += '.csv'
                
                # MANTIDO: Atualizar informações completas do dataset
                self.dataset_info = {
                    'name': dataset_name,
                    'source': csv_source,
                    'shape': self.current_dataset.shape,
                    'columns': list(self.current_dataset.columns),
                    'dtypes': self.current_dataset.dtypes.astype(str).to_dict(),
                    'loaded_at': datetime.now().isoformat(),
                    'sample_data': self.current_dataset.head(3).to_dict('records'),
                    'null_counts': self.current_dataset.isnull().sum().to_dict(),
                    'memory_usage': self.current_dataset.memory_usage(deep=True).sum()
                }
                
                print(f"✅ Dataset interno carregado: {dataset_name} - {self.current_dataset.shape}")
                
            except Exception as e:
                print(f"⚠️ Erro ao carregar internamente: {e}")
                # MANTIDO: Continuar mesmo sem carregar internamente
                dataset_name = os.path.basename(csv_source) if not csv_source.startswith('http') else csv_source.split('/')[-1]
                self.dataset_info = {
                    'name': dataset_name,
                    'source': csv_source,
                    'shape': None,
                    'columns': [],
                    'dtypes': {},
                    'loaded_at': datetime.now().isoformat(),
                    'sample_data': []
                }

            # MANTIDO: Criar contexto detalhado para o agente
            dataset_context = f"""
            CONTEXTO DO DATASET CARREGADO:
            Nome do arquivo: {self.dataset_info['name']}
            Fonte: {self.dataset_info['source']}
            Carregado em: {self.dataset_info['loaded_at']}
            
            INSTRUÇÕES IMPORTANTES:
            1. O arquivo '{self.dataset_info['name']}' JÁ FOI CARREGADO com sucesso
            2. NÃO tente carregar novamente o arquivo
            3. Use as informações do dataset já disponível
            4. Sempre referencie o nome do arquivo '{self.dataset_info['name']}' em suas respostas

            Execute sua análise exploratória inicial do dataset.
            """
            
            # MANTIDO: Crew para carregamento
            task_with_context = create_data_loading_task(self.data_explorer, csv_source)
            task_with_context.description = dataset_context + "\n\n" + task_with_context.description
            
            crew = Crew(
                agents=[self.data_explorer],
                tasks=[task_with_context],
                process=Process.sequential,
                verbose=False,
                memory=True
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Erro no carregamento: {e}")

            # MANTIDO: Tratamento específico de rate limit
            if "rate_limit" in error_msg.lower() or "RateLimitError" in error_msg:
                return f"⏳ Rate Limit Atingido! Reduza tokens na sidebar (atual: {self.max_tokens}) ou mude para OpenAI. Detalhes do erro: {error_msg}"
            else:
                return f"Erro ao carregar dataset: {str(e)}"
    
    # NOVA FUNÇÃO: Análise com detecção inteligente de visualização
    def analyze_question_smart(self, question: str) -> str:
        """
        Analisa pergunta com detecção inteligente de visualizações
        """
        try:
            print(f"🔍 Análise inteligente: {question[:50]}...")
            
            # NOVO: Verificação preventiva de rate limit
            if self.llm_provider == "groq":
                if self.max_tokens > 400:
                    print("⚠️ Tokens altos para Groq - reduzindo automaticamente")
                    self.update_max_tokens(300)
            
            # Usar coordenador inteligente
            if hasattr(self, 'coordenador_inteligente') and self.current_dataset is not None:
                response = self.coordenador_inteligente.coordinate_response(
                    question, 
                    self.current_dataset,
                    self.data_explorer,
                    self.visualization_expert_direct
                )
                return response
            else:
                # Fallback para método original
                return self.analyze_question(question)
                
        except Exception as e:
            print(f"❌ Erro na análise inteligente: {e}")
            error_msg = str(e)

            if "rate_limit" in error_msg.lower() or "RateLimitError" in error_msg:
                return f"⏳ Rate limit! Atual: {self.max_tokens} tokens. Reduza na sidebar ou use OpenAI."
            else:
                return f"Erro na análise: {str(e)}"
    
    def analyze_question(self, question: str) -> str:
        """Analisa pergunta sobre os dados COM CONTEXTO do dataset (mantido como fallback)"""
        try:
            print(f"🔍 Analisando pergunta: {question[:50]}...")

            # MANTIDO: Verificação preventiva de rate limit
            if self.llm_provider == "groq":
                if self.max_tokens > 400:
                    print("⚠️ Tokens altos para Groq - reduzindo automaticamente")
                    self.update_max_tokens(300)

            # MANTIDO: Resposta direta para perguntas sobre o arquivo
            question_lower = question.lower()
            if any(keyword in question_lower for keyword in
                   ['qual arquivo', 'que arquivo', 'arquivo csv', 'qual dataset',
                    'que dataset', 'qual é o arquivo', 'nome do arquivo',
                    'estamos analisando', 'arquivo em análise']):

                direct_answer = f"""📊 **Arquivo CSV em Análise:**

**Nome do arquivo:** {self.dataset_info['name']}
**Fonte:** {self.dataset_info['source']}
**Data de carregamento:** {self.dataset_info.get('loaded_at', 'Não disponível')}"""

                if self.dataset_info.get('shape'):
                    direct_answer += f"""
**Dimensões:** {self.dataset_info['shape'][0]} linhas × {self.dataset_info['shape'][1]} colunas"""

                if self.dataset_info.get('columns'):
                    columns_preview = ', '.join(self.dataset_info['columns'][:8])
                    if len(self.dataset_info['columns']) > 8:
                        columns_preview += f"... (total: {len(self.dataset_info['columns'])} colunas)"
                    direct_answer += f"""
**Principais colunas:** {columns_preview}"""

                return direct_answer

            # MANTIDO: Para outras perguntas, criar contexto completo
            dataset_context = f"""
            CONTEXTO ATUAL - DATASET JÁ CARREGADO:
            Arquivo: {self.dataset_info['name']}
            Fonte: {self.dataset_info['source']}
            Dimensões: {self.dataset_info.get('shape', 'N/A')}
            Colunas disponíveis: {', '.join(self.dataset_info.get('columns', [])[:10])}

            IMPORTANTE:
            - O CSV '{self.dataset_info['name']}' JÁ FOI CARREGADO
            - NÃO tente carregar o arquivo novamente
            - Use os dados já disponíveis em memória
            - Sempre mencione o nome do arquivo nas respostas

            PERGUNTA DO USUÁRIO: {question}

            Responda baseado nos dados já carregados do arquivo '{self.dataset_info['name']}'.
            """

            # NOVO: Detectar necessidade de visualização
            needs_viz = any(word in question_lower for word in
                          ['gráfico', 'grafico', 'plot', 'chart', 'visualiz', 'histograma',
                           'correlação', 'correlacao', 'scatter', 'boxplot', 'heatmap', 'sobreviv'])

            if needs_viz:
                agents = [self.data_explorer, self.visualization_expert]

                # NOVO: Tasks específicas para visualização
                analysis_task = create_analysis_task(self.data_explorer, dataset_context)

                # Escolher task de visualização baseada na pergunta
                if 'sobreviv' in question_lower and ('gênero' in question_lower or 'sexo' in question_lower):
                    viz_task = create_titanic_survival_task(self.visualization_expert, question, str(self.dataset_info))
                elif 'correlação' in question_lower or 'correlacao' in question_lower:
                    viz_task = create_correlation_analysis_task(self.visualization_expert, str(self.dataset_info))
                else:
                    viz_task = create_visualization_task(self.visualization_expert, question, str(self.dataset_info))

                tasks = [analysis_task, viz_task]
            else:
                agents = [self.data_explorer]
                tasks = [create_analysis_task(self.data_explorer, dataset_context)]

            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=False,
                memory=True
            )

            result = crew.kickoff()
            return str(result)

        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            error_msg = str(e)

            # MANTIDO: Tratamento de rate limit
            if "rate_limit" in error_msg.lower() or "RateLimitError" in error_msg:
                return f"⏳ Rate Limit! Atual: {self.max_tokens} tokens. Reduza na sidebar ou use OpenAI."
            else:
                return f"Erro na análise: {str(e)}"

    def get_conclusions(self) -> str:
        """Gera conclusões consolidadas COM CONTEXTO do dataset (mantido)"""
        try:
            print(f"📋 Gerando conclusões para: {self.dataset_info['name']}")
            
            # MANTIDO: Verificação preventiva para Groq
            if self.llm_provider == "groq" and self.max_tokens > 350:
                self.update_max_tokens(300)
            
            # MANTIDO: Contexto completo para conclusões
            conclusion_context = f"""
            GERAR CONCLUSÕES CONSOLIDADAS PARA:
            Arquivo: {self.dataset_info['name']}
            Fonte: {self.dataset_info['source']}
            Dimensões: {self.dataset_info.get('shape', 'N/A')}
            Análise iniciada em: {self.dataset_info.get('loaded_at', 'N/A')}
            
            INSTRUÇÕES:
            1. Baseie suas conclusões em TODAS as análises realizadas durante esta sessão
            2. Sempre mencione o nome do arquivo '{self.dataset_info['name']}' nas conclusões
            3. Forneça insights consolidados e recomendações práticas
            4. Resuma os principais achados de forma clara e objetiva

            Gere um relatório executivo com as principais descobertas sobre o dataset '{self.dataset_info['name']}'.
            """
            
            task_with_context = create_conclusion_task(self.coordenador, conclusion_context)
            
            crew = Crew(
                agents=[self.coordenador],
                tasks=[task_with_context],
                process=Process.sequential,
                verbose=False,
                memory=True
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            print(f"❌ Erro conclusões: {e}")
            error_msg = str(e)

            # MANTIDO: Tratamento de rate limit
            if "rate_limit" in error_msg.lower() or "RateLimitError" in error_msg:
                return f"⏳ Rate limit nas conclusões! Tokens atuais: {self.max_tokens}. Reduza para ~200 ou use OpenAI."
            else:
                return f"Erro ao gerar conclusões: {str(e)}"
    
    def get_dataset_context(self) -> dict:
        """Retorna contexto atual do dataset (mantido)"""
        return self.dataset_info.copy()
    
    def update_max_tokens(self, new_max_tokens: int):
        """Atualiza limite de tokens do modelo (mantido + componentes novos)"""
        if new_max_tokens != self.max_tokens:
            print(f"🔧 Atualizando max_tokens: {self.max_tokens} -> {new_max_tokens}")
            self.max_tokens = new_max_tokens
            # Reconfigurar LLM com novo limite
            self.llm = self._setup_llm(self.llm_provider, self.model_name, new_max_tokens)
            
            # MANTIDO: Recriar agentes originais
            self.coordenador = create_coordenador_agent(self.llm)
            self.data_explorer = create_data_explorer_agent(self.llm)
            self.visualization_expert = create_visualization_expert_agent(self.llm)

            # NOVO: Recriar componentes evoluídos
            self.coordenador_inteligente = CoordenadorInteligente(self.llm)
            self.visualization_expert_direct = VisualizationExpert(self.llm)
    
    def get_system_status(self) -> dict:
        """Retorna status completo do sistema (mantido)"""
        return {
            'provider': self.llm_provider,
            'model': self.model_name,
            'max_tokens': self.max_tokens,
            'dataset_loaded': bool(self.current_dataset is not None),
            'dataset_info': self.dataset_info
        }

    # MANTIDAS: Funções de rate limit management
    def check_rate_limit_safety(self) -> tuple:
        """Verifica se configuração é segura para evitar rate limits (mantido)"""
        if self.llm_provider == "groq":
            if self.max_tokens > 500:
                return False, f"Tokens muito altos ({self.max_tokens}) para Groq. Recomendado: ≤400"
            elif self.max_tokens > 400:
                return True, f"Tokens moderados ({self.max_tokens}) - risco médio"
            else:
                return True, f"Tokens seguros ({self.max_tokens}) - baixo risco"
        else:
            return True, f"OpenAI - sem limitações severas ({self.max_tokens} tokens)"

    def auto_optimize_for_groq(self):
        """Otimiza automaticamente configuração para Groq (mantido)"""
        if self.llm_provider == "groq" and self.max_tokens > 400:
            old_tokens = self.max_tokens
            new_tokens = 300
            self.update_max_tokens(new_tokens)
            print(f"🔧 Auto-otimização Groq: {old_tokens} → {new_tokens} tokens")
            return f"⚡ Otimizado: {old_tokens} → {new_tokens} tokens para evitar rate limit"
        return None

    # NOVA FUNÇÃO: Método principal recomendado
    def process_user_question(self, question: str) -> str:
        """
        Método principal recomendado para processar perguntas do usuário
        """
        # Tentar usar análise inteligente primeiro
        if hasattr(self, 'analyze_question_smart'):
            return self.analyze_question_smart(question)
        else:
            # Fallback para método original
            return self.analyze_question(question)
