# Data Scout — Plataforma de EDA Automatizada

Uma aplicação para realizar análise exploratória de dados de forma assistida por agentes e gerar visualizações automaticamente.

## O que o projeto oferece

- Módulos agentes para coordenação, exploração de dados e geração de visualizações.
- Análises EDA automatizadas que usam modelos de linguagem para extrair insights.
- Geração automática de gráficos e painéis a partir de comandos em linguagem natural.
- Memória simples para manter contexto entre operações durante uma sessão.
- Interface web via Streamlit para interação por chat.
- Compatibilidade com múltiplos provedores de LLM (por exemplo, Groq, OpenAI e Gemini).

## Dependências mínimas

- Python 3.11 ou superior
- Chaves de API para os provedores de LLM que for usar (Groq / OpenAI / Gemini / Google)
- Streamlit instalado no ambiente
- O pacote CrewAI (quando aplicável)

## Como preparar o ambiente

1. Faça um clone do repositório e entre na pasta do projeto:

```bash
git clone <seu-repo>
cd EDA_Agent_I2A2
```

2. Crie e ative um ambiente virtual (recomendado):

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

3. Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

4. Copie o arquivo de exemplo de variáveis de ambiente e atualize com suas chaves:

```bash
cp .env.example .env
# Edite .env e preencha suas chaves
```

5. Inicie a interface localmente:

```bash
streamlit run streamlit_app.py
```

## Publicação / Deploy

Opções comuns para colocar a aplicação no ar:

- Railway: conecte seu repositório, defina as variáveis de ambiente e permita o deploy automático.
- Docker: crie uma imagem e execute um container com o arquivo `.env`:

```bash
docker build -t eda-agent .
docker run -p 8501:8501 --env-file .env eda-agent
```

## Exemplos práticos de comandos

- "Liste os tipos de cada coluna" — resumo dos dtypes
- "Mostre histograma/boxplot para X" — visualização de distribuição
- "Calcule matriz de correlação" — análise de relacionamento entre variáveis
- "Detecte possíveis outliers" — resumo e marcação de observações atípicas
- "Gere um gráfico de dispersão entre A e B" — plot para correlação/inspeção visual

## Organização do repositório

Pasta principal do projeto:

```
EDA_Agent_I2A2/
├─ agents/        # lógica dos agentes
├─ tasks/         # módulos que representam tarefas específicas
├─ tools/         # helpers e geradores de gráficos
├─ utils/         # utilitários e configuração
├─ streamlit_app.py
└─ main.py        # integração com CrewAI
```

## Como contribuir

1. Faça um fork do repositório
2. Crie uma branch para suas alterações
3. Faça commits claros e atômicos
4. Envie a branch para seu fork e abra um Pull Request

## Licença

Este projeto está licenciado sob MIT.

---

Observação: para encerrar uma sessão interativa, digite 'f', 'F', 'fim' ou 'Fim'.

## Integração com Gemini

O projeto pode se conectar a modelos Gemini; abaixo estão orientações práticas para configurar e testar essa integração:

1) Obtenha uma chave de API

 - Registre-se no provedor Gemini/Google Cloud e gere uma chave de acesso.

2) Variáveis de ambiente recomendadas

Adicione estas entradas ao seu `.env`:

```
GEMINI_API_KEY=seu_token_aqui
GEMINI_PROJECT_ID=opcional_projeto
GEMINI_ENDPOINT=https://api.gemini.example.com
```

3) Cliente de exemplo em Python

Use a biblioteca requests para testar chamadas simples:

```python
import os
import requests

API_KEY = os.getenv('GEMINI_API_KEY')
ENDPOINT = os.getenv('GEMINI_ENDPOINT')

def call_gemini(prompt: str, max_tokens: int = 512) -> str:
	headers = {
		'Authorization': f'Bearer {API_KEY}',
		'Content-Type': 'application/json'
	}
	body = {'prompt': prompt, 'max_tokens': max_tokens}
	r = requests.post(f"{ENDPOINT}/v1/generate", json=body, headers=headers, timeout=30)
	r.raise_for_status()
	resp = r.json()
	# Ajuste conforme o formato real de retorno
	return resp.get('text') or resp.get('output', '')

```

4) Uso dentro dos agentes

- Troque o adaptador de LLM atual pela função cliente acima ou implemente um wrapper que escolha o provedor conforme configuração.
- Ajuste parâmetros como timeout, número máximo de tokens e parsing da resposta para o formato que o Gemini retornar.

5) Cuidados e custos

- Nunca comite chaves em repositórios públicos.
- Monitore consumo e limites para evitar custos inesperados.
- Faça testes com prompts curtos antes de rodar grandes lotes.
