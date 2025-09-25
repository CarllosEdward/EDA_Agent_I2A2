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
    Ferramenta para gerenciar memória e histórico de análises.
    Armazena descobertas, conclusões e contexto entre perguntas.
    OTIMIZADA: Mantém apenas informações essenciais para economizar recursos.
    """
    
    # Declarar memory_file como Field do Pydantic
    memory_file: str = Field(default="", description="Caminho do arquivo de memória")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Configurar memory_file após inicialização
        if not self.memory_file:
            self.memory_file = os.path.join(Config.TEMP_DIR, "analysis_memory.json")
        self._ensure_memory_file()
    
    def _ensure_memory_file(self):
        """Garante que o arquivo de memória existe"""
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        if not os.path.exists(self.memory_file):
            self._save_memory({
                "session_start": datetime.now().isoformat(),
                "analyses": [],
                "conclusions": [],
                "dataset_info": {},
                "insights": []
            })
    
    def _run(self, action: str, data: Optional[str] = None) -> str:
        """
        Executa ações de gerenciamento de memória OTIMIZADAS
        Args:
            action: 'save', 'get', 'get_summary', 'clear', 'add_conclusion'
            data: dados a serem salvos (JSON string)
        """
        try:
            # Garantir que memory_file está configurado
            if not hasattr(self, 'memory_file') or not self.memory_file:
                self.memory_file = os.path.join(Config.TEMP_DIR, "analysis_memory.json")
                self._ensure_memory_file()
            
            if action == "save" and data:
                memory_data = json.loads(data)
                self._add_analysis(memory_data)
                return "Análise salva na memória"
            
            elif action == "get":
                memory = self._load_memory()
                return json.dumps(memory, indent=2)
            
            # NOVA FUNÇÃO: Resumo otimizado
            elif action == "get_summary":
                return self._get_optimized_summary()
            
            elif action == "clear":
                self._clear_memory()
                return "Memória limpa"
            
            elif action == "add_conclusion" and data:
                self._add_conclusion(data)
                return "Conclusão adicionada à memória"
            
            else:
                return f"Ação não reconhecida: {action}"
                
        except Exception as e:
            return f"Erro no gerenciamento de memória: {str(e)}"
    
    def _get_optimized_summary(self) -> str:
        """NOVA FUNÇÃO: Retorna resumo otimizado da sessão (últimos 3 itens apenas)"""
        memory = self._load_memory()
        
        summary = []
        
        # Dataset info (se disponível)
        if memory.get("dataset_info"):
            summary.append(f"📊 Dataset: {memory['dataset_info']}")
        
        # Últimas 3 análises apenas
        recent_analyses = memory.get("analyses", [])[-3:]
        if recent_analyses:
            summary.append("📝 Análises recentes:")
            for i, analysis in enumerate(recent_analyses, 1):
                # Truncar para 100 caracteres
                content = str(analysis).replace('\n', ' ')[:100] + "..."
                summary.append(f"  {i}. {content}")
        
        # Última conclusão apenas
        conclusions = memory.get("conclusions", [])
        if conclusions:
            last_conclusion = conclusions[-1]["conclusion"][:150] + "..."
            summary.append(f"🎯 Última conclusão: {last_conclusion}")
        
        return "\n".join(summary) if summary else "📭 Sessão iniciada recentemente"
    
    def _load_memory(self) -> Dict[str, Any]:
        """Carrega memória do arquivo"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Erro ao carregar memória: {e}")
        
        # Retornar estrutura padrão se houver erro
        return {
            "session_start": datetime.now().isoformat(),
            "analyses": [],
            "conclusions": [],
            "dataset_info": {},
            "insights": []
        }
    
    def _save_memory(self, memory_data: Dict[str, Any]):
        """Salva memória no arquivo COM LIMITE DE ITENS"""
        try:
            # Garantir que o diretório existe
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # OTIMIZAÇÃO: Limitar número de itens para economizar espaço
            if "analyses" in memory_data and len(memory_data["analyses"]) > 10:
                memory_data["analyses"] = memory_data["analyses"][-10:]  # Últimos 10
            
            if "conclusions" in memory_data and len(memory_data["conclusions"]) > 5:
                memory_data["conclusions"] = memory_data["conclusions"][-5:]  # Últimas 5
            
            memory_data["last_updated"] = datetime.now().isoformat()
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar memória: {e}")
    
    def _add_analysis(self, analysis_data: Dict[str, Any]):
        """Adiciona nova análise à memória COM TRUNCAMENTO"""
        memory = self._load_memory()
        
        # OTIMIZAÇÃO: Truncar dados muito longos
        analysis_str = str(analysis_data)
        if len(analysis_str) > 1000:
            analysis_data = {"summary": analysis_str[:1000] + "... [truncado]", "timestamp": datetime.now().isoformat()}
        else:
            analysis_data["timestamp"] = datetime.now().isoformat()
        
        memory["analyses"].append(analysis_data)
        self._save_memory(memory)
    
    def _add_conclusion(self, conclusion: str):
        """Adiciona conclusão à memória COM TRUNCAMENTO"""
        memory = self._load_memory()
        
        # OTIMIZAÇÃO: Truncar conclusões muito longas
        if len(conclusion) > 500:
            conclusion = conclusion[:500] + "... [truncado]"
        
        conclusion_data = {
            "conclusion": conclusion,
            "timestamp": datetime.now().isoformat()
        }
        memory["conclusions"].append(conclusion_data)
        self._save_memory(memory)
    
    def get_previous_analyses(self) -> List[Dict[str, Any]]:
        """Retorna análises anteriores (máximo 5 mais recentes)"""
        memory = self._load_memory()
        return memory.get("analyses", [])[-5:]  # OTIMIZAÇÃO: Últimas 5 apenas
    
    def get_conclusions(self) -> List[str]:
        """Retorna conclusões anteriores (máximo 3 mais recentes)"""
        memory = self._load_memory()
        conclusions = memory.get("conclusions", [])[-3:]  # OTIMIZAÇÃO: Últimas 3 apenas
        return [c["conclusion"] for c in conclusions]
    
    def _clear_memory(self):
        """Limpa a memória"""
        self._save_memory({
            "session_start": datetime.now().isoformat(),
            "analyses": [],
            "conclusions": [],
            "dataset_info": {},
            "insights": []
        })

