import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from crewai.tools import BaseTool
from pydantic import Field
from utils.config import Config

class MemoryManagerTool(BaseTool):
    name: str = "Memory Manager"
    description: str = """
    Ferramenta para gerenciar a memória e o histórico de análises de uma sessão.
    Armazena descobertas e conclusões para manter o contexto.
    """
    
    # Define o campo para o arquivo de memória, com um valor padrão.
    memory_file: str = Field(default="", description="Caminho do arquivo de memória")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Configura o caminho do arquivo de memória e garante que ele existe.
        if not self.memory_file:
            self.memory_file = os.path.join(Config.TEMP_DIR, "analysis_memory.json")
        self._ensure_memory_file()
    
    def _ensure_memory_file(self):
        """
        Garante que o arquivo de memória e o diretório existem.
        Se o arquivo não existir, cria uma estrutura inicial vazia para a sessão.
        """
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        if not os.path.exists(self.memory_file):
            self._save_memory({
                "session_start": datetime.now().isoformat(),
                "analyses": [],
                "conclusions": [],
                "dataset_info": {},
            })
    
    def _run(self, action: str, data: Optional[str] = None) -> str:
        """
        Executa as ações de gerenciamento de memória solicitadas pelos agentes.
        Este método é a interface principal da ferramenta.
        """
        try:
            if not hasattr(self, 'memory_file') or not self.memory_file:
                self.memory_file = os.path.join(Config.TEMP_DIR, "analysis_memory.json")
                self._ensure_memory_file()
            
            # Chama a função de salvamento de análise se a ação for 'save'
            if action == "save" and data:
                memory_data = json.loads(data)
                self._add_analysis(memory_data)
                return "Análise salva na memória."
            
            # Retorna todo o conteúdo da memória, usado para debug ou análise profunda
            elif action == "get":
                memory = self._load_memory()
                return json.dumps(memory, indent=2)
            
            # Retorna um resumo otimizado, evitando sobrecarga de dados para o LLM
            elif action == "get_summary":
                return self._get_optimized_summary()
            
            # Limpa a memória, iniciando uma nova sessão
            elif action == "clear":
                self._clear_memory()
                return "Memória limpa."
            
            # Adiciona uma conclusão consolidada à memória
            elif action == "add_conclusion" and data:
                self._add_conclusion(data)
                return "Conclusão adicionada à memória."
            
            else:
                return f"Ação não reconhecida: {action}"
                
        except Exception as e:
            return f"Erro no gerenciamento de memória: {str(e)}"
    
    def _get_optimized_summary(self) -> str:
        """
        Gera um resumo da sessão de análise.
        Esta função é otimizada para o LLM, retornando apenas as informações mais
        relevantes e recentes, como o resumo do dataset e as últimas análises e conclusões.
        """
        memory = self._load_memory()
        
        summary = []
        
        # Adiciona informações do dataset se estiverem disponíveis
        if memory.get("dataset_info"):
            summary.append(f"📊 Dataset: {memory['dataset_info']}")
        
        # Inclui apenas as últimas 3 análises para manter o resumo conciso
        recent_analyses = memory.get("analyses", [])[-3:]
        if recent_analyses:
            summary.append("📝 Análises recentes:")
            for i, analysis in enumerate(recent_analyses, 1):
                content = str(analysis).replace('\n', ' ')[:100] + "..."
                summary.append(f"  {i}. {content}")
        
        # Inclui apenas a última conclusão
        conclusions = memory.get("conclusions", [])
        if conclusions:
            last_conclusion = conclusions[-1]["conclusion"][:150] + "..."
            summary.append(f"🎯 Última conclusão: {last_conclusion}")
        
        return "\n".join(summary) if summary else "📭 Sessão de análise iniciada recentemente."
    
    def _load_memory(self) -> Dict[str, Any]:
        """
        Carrega os dados de memória do arquivo JSON.
        Inclui um tratamento de erro para garantir que a aplicação continue
        mesmo se o arquivo de memória estiver corrompido ou inacessível.
        """
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Erro ao carregar memória: {e}")
        
        return {
            "session_start": datetime.now().isoformat(),
            "analyses": [],
            "conclusions": [],
            "dataset_info": {},
        }
    
    def _save_memory(self, memory_data: Dict[str, Any]):
        """
        Salva os dados de memória no arquivo, com otimizações.
        Limita o número de análises e conclusões para evitar que o arquivo
        cresça indefinidamente, o que é crucial para a performance.
        """
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            if "analyses" in memory_data and len(memory_data["analyses"]) > 10:
                memory_data["analyses"] = memory_data["analyses"][-10:]
            
            if "conclusions" in memory_data and len(memory_data["conclusions"]) > 5:
                memory_data["conclusions"] = memory_data["conclusions"][-5:]
            
            memory_data["last_updated"] = datetime.now().isoformat()
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar memória: {e}")
    
    def _add_analysis(self, analysis_data: Dict[str, Any]):
        """
        Adiciona uma nova análise à memória, truncando-a se for muito longa
        para evitar sobrecarga de dados.
        """
        memory = self._load_memory()
        
        analysis_str = str(analysis_data)
        if len(analysis_str) > 1000:
            analysis_data = {"summary": analysis_str[:1000] + "... [truncado]", "timestamp": datetime.now().isoformat()}
        else:
            analysis_data["timestamp"] = datetime.now().isoformat()
        
        memory["analyses"].append(analysis_data)
        self._save_memory(memory)
    
    def _add_conclusion(self, conclusion: str):
        """
        Adiciona uma nova conclusão à memória, também truncando-a para
        evitar que se torne muito grande.
        """
        memory = self._load_memory()
        
        if len(conclusion) > 500:
            conclusion = conclusion[:500] + "... [truncado]"
        
        conclusion_data = {
            "conclusion": conclusion,
            "timestamp": datetime.now().isoformat()
        }
        memory["conclusions"].append(conclusion_data)
        self._save_memory(memory)
    
    def get_previous_analyses(self) -> List[Dict[str, Any]]:
        """
        Retorna as últimas 5 análises para fornecer contexto relevante sem
        sobrecarregar a memória do agente.
        """
        memory = self._load_memory()
        return memory.get("analyses", [])[-5:]
    
    def get_conclusions(self) -> List[str]:
        """
        Retorna as últimas 3 conclusões para manter o contexto principal da sessão.
        """
        memory = self._load_memory()
        conclusions = memory.get("conclusions", [])[-3:]
        return [c["conclusion"] for c in conclusions]
    
    def _clear_memory(self):
        """
        Limpa a memória, recriando a estrutura inicial da sessão.
        """
        self._save_memory({
            "session_start": datetime.now().isoformat(),
            "analyses": [],
            "conclusions": [],
            "dataset_info": {},
        })
