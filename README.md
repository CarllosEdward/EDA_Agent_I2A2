# 🤖 EDA Agente Inteligente

Sistema de Análise Exploratória de Dados usando Agentes Inteligentes com CrewAI.

## 🚀 Características

- **Agentes Especializados**: Coordenador, Explorador de Dados, Especialista em Visualização
- **Análise Inteligente**: EDA automatizada com insights baseados em LLM
- **Visualizações**: Gráficos automáticos baseados nas perguntas
- **Memória Persistente**: Mantém contexto entre análises
- **Interface Amigável**: Chat interativo com Streamlit
- **Múltiplos LLMs**: Suporte para Groq e OpenAI

## 📋 Requisitos

- Python 3.11+
- API Key do Groq ou OpenAI
- Streamlit
- CrewAI

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone <seu-repo>
cd projeto-eda-csv
```

2. Crie ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale dependências:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
```bash
cp .env.example .env
# Edite o .env com suas chaves
```

5. Execute localmente:
```bash
streamlit run streamlit_app.py
```

## 🌐 Deploy

### Railway
1. Conecte seu repositório GitHub ao Railway
2. Configure as variáveis de ambiente
3. Deploy automático

### Docker
```bash
docker build -t eda-agent .
docker run -p 8501:8501 --env-file .env eda-agent
```

## 📊 Exemplos de Uso

- "Quais são os tipos de dados?"
- "Mostre a distribuição das variáveis"
- "Existe correlação entre as colunas?"
- "Identifique outliers nos dados"
- "Crie um gráfico de correlação"
- "Quais suas conclusões dos dados?"

## 🔧 Estrutura do Projeto

```
projeto-eda-csv/
├── agents/          # Agentes especializados
├── tasks/           # Tarefas específicas
├── tools/           # Ferramentas customizadas
├── utils/           # Utilitários
├── streamlit_app.py # Interface principal
└── main.py          # Sistema CrewAI
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

MIT License

---

**Finalização:** Digite 'f', 'F', 'fim' ou 'Fim' para encerrar sessões.
