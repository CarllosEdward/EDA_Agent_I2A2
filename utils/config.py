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
    
    # Chaves da OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # NOVAS CHAVES E DEPENDÊNCIAS PARA O GEMINI
    # Adicione a variável de ambiente para a chave da API do Google
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # Adicione a variável para o modelo do Gemini, se quiser especificar um
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # App Configuration
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "128"))  # MB
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
        # Inclui a validação para a nova chave do Gemini
        if not cls.GOOGLE_API_KEY:
            issues.append("GOOGLE_API_KEY não configurada")
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
            "gemini_configured": bool(cls.GOOGLE_API_KEY), # Adicionado: status da chave do Gemini
            "port": cls.PORT
        }
