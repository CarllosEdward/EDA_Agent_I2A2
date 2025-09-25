FROM python:3.13-slim

WORKDIR /app

# Instalar dependências do sistema necessárias para compilação
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash app

# Copiar requirements primeiro (para melhor cache de layers Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo o código da aplicação
COPY . .

# Criar diretórios necessários com permissões corretas
RUN mkdir -p temp_files/uploads temp_files/charts temp_files/memory && \
    chown -R app:app /app

# Mudar para usuário não-root
USER app

# Railway define PORT automaticamente via variável de ambiente
EXPOSE $PORT

# Comando otimizado para Railway
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.runOnSave=false --browser.serverAddress=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
