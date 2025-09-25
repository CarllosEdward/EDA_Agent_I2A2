import os
from dotenv import load_dotenv

# Carregar .env apenas se existir (localhost)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("📁 Carregado arquivo .env local")
else:
    print("☁️ Usando variáveis de ambiente Railway")

class Config:
    # LLM Configuration - Funciona com Railway e localhost
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # App Configuration
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50"))  # MB
    TEMP_DIR = os.getenv("TEMP_DIR", "temp_files")
    UPLOAD_DIR = f"{TEMP_DIR}/uploads"
    
    # Railway Configuration
    PORT = int(os.getenv("PORT", "8501"))  # Railway define PORT automaticamente
    
    # Streamlit Configuration
    PAGE_TITLE = "EDA Agente Inteligente"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    @classmethod
    def validate_keys(cls):
        """Valida se as chaves API estão configuradas"""
        issues = []
        if not cls.GROQ_API_KEY:
            issues.append("GROQ_API_KEY não configurada")
        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY não configurada")
        return issues
    
    @classmethod
    def get_environment(cls):
        """Detecta se está rodando no Railway ou localhost"""
        railway_vars = ['RAILWAY_PROJECT_ID', 'RAILWAY_SERVICE_ID', 'PORT']
        is_railway = any(os.getenv(var) for var in railway_vars)
        return "railway" if is_railway else "localhost"
    
    @classmethod
    def setup_directories(cls):
        """Configura diretórios necessários"""
        directories = [
            cls.TEMP_DIR,
            cls.UPLOAD_DIR,
            os.path.join(cls.TEMP_DIR, "charts"),
            os.path.join(cls.TEMP_DIR, "memory")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    @classmethod
    def get_status(cls):
        """Status da configuração"""
        env = cls.get_environment()
        return {
            "environment": env,
            "groq_configured": bool(cls.GROQ_API_KEY),
            "openai_configured": bool(cls.OPENAI_API_KEY),
            "port": cls.PORT
        }

