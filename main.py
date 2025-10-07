import os
import pandas as pd
from crewai import Crew, Process
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from utils.config import Config
from agents import create_data_explorer_agent
from agents.coordenador import CoordenadorInteligente
from agents.visualization_expert import VisualizationExpert
from tasks import create_data_loading_task, create_conclusion_task
from utils.helpers import ensure_directories
from datetime import datetime

class EDACrewSystem:
    """Sistema principal de análise EDA com CrewAI."""
    
    def __init__(self, llm_provider: str = "openai", model_name: str = None, max_tokens: int = 1500):
        """
        Inicializa o sistema EDA.
        Args:
            llm_provider: 'groq' ou 'openai'.
            model_name: Nome específico do modelo.
            max_tokens: Limite de tokens para respostas.
        """
        print(f"🔧 Inicializando EDACrewSystem: {llm_provider} - {model_name} (max_tokens: {max_tokens})")
        
        ensure_directories()
        self.llm_provider = llm_provider
        self.model_name = model_name or self._get_default_model(llm_provider)
        self.max_tokens = max_tokens
        self.llm = self._setup_llm(llm_provider, self.model_name, max_tokens)
        
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
        
        # Inicializa agentes principais
        self.data_explorer = create_data_explorer_agent(self.llm)
        self.coordinator = CoordenadorInteligente(self.llm)
        self.visualization_expert = VisualizationExpert(self.llm)
    
    def _get_default_model(self, provider: str) -> str:
        """Retorna modelo padrão para o provider."""
        defaults = {
            "groq": "llama-3.1-8b-instant",
            "openai": "gpt-3.5-turbo"
        }
        return defaults.get(provider.lower(), "gpt-3.5-turbo")
    
    def _setup_llm(self, provider: str, model_name: str, max_tokens: int):
        """Configura o modelo LLM."""
        print(f"🔧 Configurando LLM: {provider} - {model_name} (tokens: {max_tokens})")
        
        if provider.lower() == "groq":
            api_key = Config.GROQ_API_KEY
            if not api_key:
                raise ValueError("GROQ_API_KEY não configurada")
            
            if not any(model_name.startswith(prefix) for prefix in ["groq/", "openai/", "qwen/", "moonshotai/"]):
                groq_model = f"groq/{model_name}"
            else:
                groq_model = model_name
                
            print(f"🔧 Modelo Groq configurado: {groq_model}")
            
            safe_max_tokens = min(max_tokens, 500)
            if safe_max_tokens != max_tokens:
                print(f"⚠️ Tokens reduzidos para Groq: {max_tokens} → {safe_max_tokens}")
            
            return ChatGroq(
                api_key=api_key,
                model=groq_model,
                temperature=0.1,
                max_tokens=safe_max_tokens,
                timeout=30,
                max_retries=1,
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
                max_tokens=max_tokens,
                timeout=60,
                max_retries=2
            )
        
        else:
            raise ValueError("Provider deve ser 'groq' ou 'openai'")
    
    def load_dataset(self, csv_source: str) -> str:
        """Carrega um dataset CSV e executa a análise inicial."""
        try:
            print(f"📄 Carregando: {csv_source} com {self.llm_provider}-{self.model_name}")
            
            if self.llm_provider == "groq" and self.max_tokens > 400:
                print("⚠️ Groq com tokens altos - reduzindo para evitar rate limit")
                self.update_max_tokens(400)
            
            try:
                if csv_source.startswith(('http://', 'https://')):
                    from utils.helpers import download_csv_from_url
                    temp_path = download_csv_from_url(csv_source)
                    self.current_dataset = pd.read_csv(temp_path)
                    os.unlink(temp_path)
                else:
                    self.current_dataset = pd.read_csv(csv_source)
                
                if csv_source.startswith(('http://', 'https://')):
                    dataset_name = csv_source.split('/')[-1]
                    if '?' in dataset_name:
                        dataset_name = dataset_name.split('?')[0]
                else:
                    dataset_name = os.path.basename(csv_source)
                
                if not dataset_name.endswith('.csv'):
                    if '.' not in dataset_name:
                        dataset_name += '.csv'
                
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
                dataset_name = os.path.basename(csv_source) if not csv_source.startswith('http') else csv_source.split('/')[-1]
                self.dataset_info = { 'name': dataset_name, 'source': csv_source, 'loaded_at': datetime.now().isoformat() }

            dataset_context = f"""
            CONTEXTO DO DATASET CARREGADO:
            Nome do arquivo: {self.dataset_info['name']}
            Fonte: {self.dataset_info['source']}
            Carregado em: {self.dataset_info['loaded_at']}
            
            INSTRUÇÕES: Execute sua análise exploratória inicial do dataset.
            O arquivo '{self.dataset_info['name']}' JÁ FOI CARREGADO.
            NÃO tente carregar o arquivo novamente.
            """
            
            task = create_data_loading_task(self.data_explorer, dataset_context)
            
            crew = Crew(
                agents=[self.data_explorer],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
                memory=True
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Erro no carregamento: {e}")
            if "rate_limit" in error_msg.lower() or "RateLimitError" in error_msg:
                return f"⏳ Rate Limit Atingido! Reduza tokens na sidebar (atual: {self.max_tokens}) ou mude para OpenAI. Detalhes: {error_msg}"
            else:
                return f"Erro ao carregar dataset: {str(e)}"
    
    def analyze_question(self, question: str) -> str:
        """
        Analisa uma pergunta do usuário usando o coordenador inteligente.
        """
        try:
            print(f"🔍 Análise da pergunta: {question[:50]}...")
            
            if self.llm_provider == "groq" and self.max_tokens > 400:
                print("⚠️ Tokens altos para Groq - reduzindo automaticamente")
                self.update_max_tokens(300)
            
            if self.current_dataset is not None:
                response = self.coordinator.coordinate_response(
                    question, 
                    self.current_dataset,
                    self.data_explorer,
                    self.visualization_expert
                )
                return response
            else:
                return "Erro: Nenhum dataset carregado. Por favor, carregue um arquivo primeiro."
                
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "RateLimitError" in error_msg:
                return f"⏳ Rate limit! Atual: {self.max_tokens} tokens. Reduza na sidebar ou use OpenAI."
            else:
                return f"Erro na análise: {str(e)}"
    
    def get_conclusions(self) -> str:
        """Gera conclusões consolidadas sobre a análise."""
        try:
            print(f"📋 Gerando conclusões para: {self.dataset_info['name']}")
            
            if self.llm_provider == "groq" and self.max_tokens > 350:
                self.update_max_tokens(300)
            
            conclusion_context = f"""
            GERAR CONCLUSÕES CONSOLIDADAS PARA:
            Arquivo: {self.dataset_info['name']}
            Dimensões: {self.dataset_info.get('shape', 'N/A')}
            
            INSTRUÇÕES:
            Baseie suas conclusões em TODAS as análises realizadas nesta sessão.
            Forneça insights consolidados e resuma os principais achados.
            """
            
            task = create_conclusion_task(self.coordinator, conclusion_context)
            
            crew = Crew(
                agents=[self.coordinator],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
                memory=True
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            print(f"❌ Erro nas conclusões: {e}")
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "RateLimitError" in error_msg:
                return f"⏳ Rate limit nas conclusões! Tokens: {self.max_tokens}. Reduza ou use OpenAI."
            else:
                return f"Erro ao gerar conclusões: {str(e)}"
    
    def get_dataset_context(self) -> dict:
        """Retorna o contexto atual do dataset."""
        return self.dataset_info.copy()
    
    def update_max_tokens(self, new_max_tokens: int):
        """Atualiza o limite de tokens do modelo e recria os agentes."""
        if new_max_tokens != self.max_tokens:
            print(f"🔧 Atualizando max_tokens: {self.max_tokens} -> {new_max_tokens}")
            self.max_tokens = new_max_tokens
            self.llm = self._setup_llm(self.llm_provider, self.model_name, new_max_tokens)
            
            self.data_explorer = create_data_explorer_agent(self.llm)
            self.coordinator = CoordenadorInteligente(self.llm)
            self.visualization_expert = VisualizationExpert(self.llm)
    
    def get_system_status(self) -> dict:
        """Retorna o status completo do sistema."""
        return {
            'provider': self.llm_provider,
            'model': self.model_name,
            'max_tokens': self.max_tokens,
            'dataset_loaded': bool(self.current_dataset is not None),
            'dataset_info': self.dataset_info
        }
