import streamlit as st
import pandas as pd
import os
import tempfile
import json
from datetime import datetime
from PIL import Image
import plotly.express as px
from utils.config import Config
from utils.helpers import ensure_directories, clean_temp_files, validate_csv_file
from main import EDACrewSystem

# Configuração da página
st.set_page_config(
    page_title="EDA Agente Inteligente",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background-color: #f0f2f6;
    }
    .agent-response {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        background-color: #d4edda;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d1edff;
        border: 1px solid #0066cc;
        margin: 1rem 0;
    }
    .dataset-info {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e8f4fd;
        border: 1px solid #0066cc;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa o estado da sessão"""
    if 'eda_system' not in st.session_state:
        st.session_state.eda_system = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_dataset_info' not in st.session_state:
        st.session_state.current_dataset_info = {}
    if 'current_config' not in st.session_state:
        st.session_state.current_config = ""
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'dataset_source' not in st.session_state:
        st.session_state.dataset_source = ""
    if 'session_finalized' not in st.session_state:
        st.session_state.session_finalized = False

def handle_rate_limit_error(error_msg: str, provider: str):
    """Função para tratar rate limits de forma inteligente"""
    if "rate limit" in error_msg.lower() or "rate_limit_exceeded" in error_msg:
        st.error("⏳ **Rate Limit Atingido!** Reduza tokens na sidebar ou mude para OpenAI.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reduzir Tokens", key="reduce_tokens"):
                if provider == "groq":
                    st.session_state.tokens_slider = 200
                    st.rerun()
        with col2:
            if st.button("🚀 Mudar OpenAI", key="switch_openai"):
                st.session_state.llm_provider_select = "openai"
                st.rerun()
        
        return True
    return False

def setup_sidebar():
    """Configura a barra lateral - VERSÃO CORRIGIDA"""
    st.sidebar.header("⚙️ Configurações do Sistema")
    
    # CORREÇÃO 1: OpenAI como padrão (mais confiável)
    llm_provider = st.sidebar.selectbox(
        "🔧 Provedor LLM:",
        ["openai", "groq"],  # MUDANÇA: OpenAI primeiro para ser padrão
        help="OpenAI é mais estável. Groq é rápido mas tem rate limits severos",
        key="llm_provider_select"
    )
    
    # CORREÇÃO 2: Modelos atualizados (válidos em 2025) - ORDEM OTIMIZADA
    model_options = {
        "groq": [
            "llama-3.3-70b-versatile",   # PRIMEIRO: Mais poderoso e atualizado
            "llama-3.1-8b-instant",      # Segundo: Mais rápido
            "openai/gpt-oss-120b",       # Terceiro: Modelo OpenAI open
            "qwen/qwen3-32b"             # Quarto: Substituiu mistral
        ],
        "openai": [
            "gpt-4o-mini",              # PRIMEIRO: Mais eficiente e barato
            "gpt-3.5-turbo",            # Segundo: Clássico confiável
            "gpt-4",                    # Terceiro: Mais poderoso
            "gpt-4-turbo-preview"       # Quarto: Mais recente
        ]
    }
    
    model_name = st.sidebar.selectbox(
        "🎯 Modelo:",
        model_options[llm_provider],
        index=0,
        help=f"Modelo {llm_provider.upper()} para usar",
        key="model_select"
    )
    
    # Configurações específicas para Groq
    if llm_provider == "groq":
        st.sidebar.warning("⚠️ Groq: Rate limits severos (6000 tokens/min)")
        
        # CORREÇÃO 3: Limite de tokens mais baixo para Groq
        max_tokens = st.sidebar.slider(
            "🔢 Limite Tokens (Resposta):",
            min_value=100,
            max_value=600,  # REDUZIDO: de 1000 para 600
            value=300,      # REDUZIDO: de 400 para 300
            step=50,
            help="⚡ GROQ: Mantenha baixo para evitar rate limits!",
            key="tokens_slider"
        )
        
        # Salvar configuração no session state
        st.session_state.max_tokens_groq = max_tokens
        
        st.sidebar.info(f"💡 Tokens: {max_tokens} (Rate limit: {6000-max_tokens} disponíveis)")
        st.sidebar.markdown("**💡 Dica:** Se der rate limit, reduza tokens ou mude para OpenAI")
        
        # Mostrar status da configuração - AVISOS MAIS CLAROS
        if max_tokens <= 300:
            st.sidebar.success("✅ Configuração conservadora (menos rate limit)")
        elif max_tokens <= 400:
            st.sidebar.info("ℹ️ Configuração balanceada")
        else:
            st.sidebar.warning("⚠️ Configuração alta (mais rate limit)")
    else:
        st.sidebar.success("✅ OpenAI: Mais estável, sem rate limits severos")
        st.sidebar.info("💰 Usa créditos pagos da sua conta")
        # Tokens padrão para OpenAI
        max_tokens = 1500
    
    # Status das API Keys
    st.sidebar.subheader("🔑 Status das Chaves")
    issues = Config.validate_keys()
    if issues:
        for issue in issues:
            st.sidebar.error(f"❌ {issue}")
        st.sidebar.markdown("""
        **⚠️ Configuração Necessária:**
        1. Edite arquivo `.env`
        2. Adicione suas API keys:
           - `GROQ_API_KEY=gsk_...`
           - `OPENAI_API_KEY=sk_...`
        3. Reinicie aplicação
        """)
    else:
        st.sidebar.success("✅ Chaves configuradas")
    
    # Status do sistema atual
    if st.session_state.get('system_initialized'):
        current = st.session_state.get('current_config', 'Não definido')
        st.sidebar.markdown(f"**🤖 Sistema Ativo:**\n`{current}`")
        
        if llm_provider == "groq":
            current_tokens = st.session_state.get('max_tokens_groq', 300)
            st.sidebar.markdown(f"**🔢 Tokens configurados:** {current_tokens}")
    
    # Mostrar info do dataset se carregado
    if st.session_state.get('dataset_loaded'):
        dataset_source = st.session_state.get('dataset_source', 'Desconhecido')
        dataset_name = os.path.basename(dataset_source) if dataset_source else 'Dataset'
        st.sidebar.markdown(f"**📊 Dataset Ativo:**\n`{dataset_name}`")
        
        # Mostrar informações básicas do dataset
        dataset_info = st.session_state.get('current_dataset_info', {})
        if dataset_info.get('shape'):
            st.sidebar.markdown(f"**📏 Dimensões:** {dataset_info['shape'][0]}×{dataset_info['shape'][1]}")
    
    return llm_provider, model_name, locals().get('max_tokens', 1500)

def display_first_rows_improved(df, num_rows=5):
    """NOVA FUNÇÃO: Mostra primeiras linhas de forma mais legível"""
    if len(df) == 0:
        st.warning("Dataset vazio!")
        return
    
    st.subheader(f"📋 Primeiras {num_rows} linhas:")
    
    # CORREÇÃO: Mostrar cada linha separadamente para melhor leitura
    for i in range(min(num_rows, len(df))):
        with st.expander(f"📄 Linha {i+1}", expanded=(i < 2)):  # Primeiras 2 expandidas
            row_data = df.iloc[i]
            
            # Criar tabela mais legível
            for col_name, value in row_data.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**{col_name}:**")
                with col2:
                    if pd.isna(value):
                        st.write("🔍 *Valor ausente*")
                    else:
                        st.write(f"`{value}`")

def load_dataset_section():
    """Seção de carregamento de dataset"""
    st.header("📁 Carregamento de Dataset")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Local", "🌐 URL Direta", "🔗 URLs Especiais"])
    
    dataset_source = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV:",
            type=['csv'],
            help="Upload de arquivo CSV do seu computador",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            temp_dir = Config.UPLOAD_DIR
            ensure_directories()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            dataset_source = temp_path
            # Mostrar nome do arquivo na caixa de texto
            st.text_input(
                "📁 Arquivo selecionado:", 
                value=uploaded_file.name, 
                disabled=True,
                key="file_display"
            )
            st.success(f"✅ Arquivo salvo: {uploaded_file.name}")
    
    with tab2:
        # CORREÇÃO: Key para detectar mudanças automaticamente
        csv_url = st.text_input(
            "URL do arquivo CSV:",
            placeholder="https://exemplo.com/dados.csv",
            help="Cole a URL direta de um arquivo CSV",
            key="csv_url_input"  # Key importante para detecção automática
        )
        
        # CORREÇÃO: Verificação mais robusta da URL
        if csv_url and csv_url.strip():
            url_clean = csv_url.strip()
            if url_clean.startswith(('http://', 'https://')):
                dataset_source = url_clean
                # Mostrar nome do arquivo extraído da URL
                url_filename = url_clean.split('/')[-1]
                if '?' in url_filename:
                    url_filename = url_filename.split('?')[0]
                if not url_filename:
                    url_filename = "arquivo.csv"
                    
                st.text_input(
                    "🌐 Arquivo da URL:", 
                    value=url_filename, 
                    disabled=True,
                    key="url_display"
                )
                st.success("✅ URL configurada e pronta para carregamento")
            else:
                st.error("❌ URL inválida! Deve começar com http:// ou https://")
    
    with tab3:
        st.markdown("""
        **🌐 URLs de Exemplo para Teste Rápido:**
        
        **📊 Datasets Públicos Populares:**
        - **Iris (Flores):** Classificação de espécies - 150 registros
        - **Tips (Gorjetas):** Análise de restaurante - 244 registros  
        - **Titanic:** Passageiros do Titanic - 891 registros
        """)
        
        example_url = st.selectbox(
            "Escolha um dataset de exemplo:",
            ["", 
             "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
             "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
             "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"],
            format_func=lambda x: {
                "": "Selecione um exemplo...",
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv": "🌸 Iris Dataset (flores)",
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv": "🍽️ Tips Dataset (gorjetas)",
                "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv": "🚢 Titanic Dataset (passageiros)"
            }.get(x, x),
            key="example_selector"
        )
        
        if example_url:
            dataset_source = example_url
            # Mostrar nome do arquivo
            example_filename = example_url.split('/')[-1]
            st.text_input(
                "🔗 Exemplo selecionado:", 
                value=example_filename, 
                disabled=True,
                key="example_display"
            )
            st.success(f"✅ Exemplo selecionado: {example_filename}")
    
    return dataset_source

def initialize_eda_system(llm_provider: str, model_name: str, max_tokens: int):
    """Inicializa sistema EDA com detecção de mudanças"""
    current_config = f"{llm_provider}-{model_name}-{max_tokens}"
    
    # Verificar se configuração mudou
    if (st.session_state.current_config != current_config or 
        not st.session_state.system_initialized):
        
        with st.spinner(f"🤖 Inicializando sistema: {llm_provider.upper()} - {model_name} (tokens: {max_tokens})"):
            try:
                # Limpar sistema anterior se existir
                if 'eda_system' in st.session_state:
                    del st.session_state.eda_system
                
                # Resetar dataset se configuração mudou
                if st.session_state.current_config != current_config:
                    if st.session_state.get('dataset_loaded'):
                        st.session_state.dataset_loaded = False
                        st.session_state.chat_history = []
                        st.warning("⚠️ Configuração do modelo mudou! Recarregue seu dataset.")
                
                # Inicializar novo sistema
                eda_system = EDACrewSystem(llm_provider, model_name, max_tokens)
                
                # Salvar no estado
                st.session_state.eda_system = eda_system
                st.session_state.current_config = current_config
                st.session_state.system_initialized = True
                
                st.success(f"✅ Sistema inicializado: {llm_provider.upper()} - {model_name}")
                return True
                
            except Exception as e:
                st.session_state.system_initialized = False
                error_msg = str(e)
                
                # Usar função de rate limit handler
                if handle_rate_limit_error(error_msg, llm_provider):
                    return False
                elif "api" in error_msg.lower() and "key" in error_msg.lower():
                    st.error(f"🔑 **Erro de API Key - {llm_provider.upper()}**\n\nVerifique se a chave está correta no arquivo .env")
                else:
                    st.error(f"❌ **Erro de inicialização {llm_provider.upper()}:**\n{error_msg}")
                
                return False
    
    return True

def process_dataset_loading(dataset_source: str):
    """Processa carregamento do dataset - VERSÃO CORRIGIDA"""
    # CORREÇÃO: Botões com textos mais apropriados
    col1, col2 = st.columns([4, 1])  # Botão principal maior
    
    with col1:
        load_button = st.button(
            "📊 Processar Carregamento",  # MUDANÇA: Texto mais apropriado para esta fase
            type="primary", 
            use_container_width=True,
            key="load_dataset_btn"
        )
    
    with col2:
        # Botão finalizar também na página de carregamento
        finalizar_load_btn = st.button(
            "✅ Finalizar", 
            key="finalizar_load", 
            use_container_width=True,
            help="Encerrar aplicação"
        )
    
    # Processar botão finalizar
    if finalizar_load_btn:
        finalize_session()
        return
    
    if load_button:
        if not st.session_state.get('system_initialized'):
            st.error("❌ Sistema não inicializado! Verifique configurações na sidebar.")
            return
        
        # Resetar estado antes do carregamento
        st.session_state.dataset_loaded = False
        st.session_state.chat_history = []
        st.session_state.session_finalized = False
        
        # Salvar fonte do dataset
        st.session_state.dataset_source = dataset_source
        
        with st.spinner("📊 Carregando e analisando dataset..."):
            try:
                # Mostrar progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📄 Inicializando carregamento...")
                progress_bar.progress(20)
                
                # Carregar dataset com o sistema EDA
                status_text.text("📥 Baixando/lendo arquivo CSV...")
                progress_bar.progress(50)
                
                result = st.session_state.eda_system.load_dataset(dataset_source)
                
                status_text.text("🧠 Executando análise inicial...")
                progress_bar.progress(80)
                
                # Verificar resultado
                has_error = any(keyword in result.lower() for keyword in 
                              ['erro', 'error', 'exception', 'rate limit', 'failed', 'badrequest'])
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if has_error:
                    # ERRO - Não marcar como carregado
                    st.session_state.dataset_loaded = False
                    
                    # Usar função de rate limit handler
                    if not handle_rate_limit_error(result, st.session_state.get('current_config', '')):
                        if "badrequest" in result.lower() and "provider not provided" in result.lower():
                            st.error("🔧 **Erro de configuração do modelo Groq.** Mude para OpenAI temporariamente.")
                        else:
                            st.error("❌ **Falha no carregamento**")
                            
                        st.markdown(f'<div class="warning-box"><strong>Detalhes do erro:</strong><br>{result}</div>', 
                                  unsafe_allow_html=True)
                    
                    # Sugestões baseadas no tipo de erro
                    if "groq" in st.session_state.get('current_config', '').lower():
                        st.info("💡 **Sugestão:** Mude para OpenAI na sidebar (mais estável) ou reduza tokens para Groq.")
                    
                else:
                    # SUCESSO - Marcar como carregado
                    st.session_state.dataset_loaded = True
                    
                    # Extrair e salvar informações do dataset
                    dataset_name = os.path.basename(dataset_source) if dataset_source else 'Dataset'
                    if dataset_source.startswith(('http://', 'https://')):
                        dataset_name = dataset_source.split('/')[-1]
                        if '?' in dataset_name:
                            dataset_name = dataset_name.split('?')[0]
                    
                    st.session_state.current_dataset_info = {
                        'name': dataset_name,
                        'source': dataset_source,
                        'loaded_at': datetime.now().isoformat(),
                        'analysis_result': result
                    }
                    
                    # Mostrar resultado
                    st.markdown('<div class="agent-response">', unsafe_allow_html=True)
                    st.markdown("**🤖 Agente Explorador de Dados:**")
                    st.markdown(result)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Salvar no histórico
                    st.session_state.chat_history.append({
                        'type': 'system',
                        'message': f'Dataset carregado: {dataset_name}',
                        'response': result,
                        'dataset_info': st.session_state.current_dataset_info
                    })
                    
                    st.success("✅ **Dataset carregado com sucesso!** Agora você pode fazer perguntas sobre os dados.")
                    st.balloons()
                    
                    # CORREÇÃO: Botão maior e texto mais apropriado
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])  # Centralizar
                    with col2:
                        if st.button(
                            "🚀 Iniciar Análise Interativa", 
                            type="primary", 
                            key="goto_chat",
                            use_container_width=True,  # Botão maior
                            help="Ir para o chat interativo com os agentes"
                        ):
                            st.rerun()
            
            except Exception as e:
                st.session_state.dataset_loaded = False
                error_msg = str(e)
                
                # Usar função de rate limit handler
                if not handle_rate_limit_error(error_msg, st.session_state.get('current_config', '')):
                    if "api" in error_msg.lower() and "key" in error_msg.lower():
                        st.error("🔑 **Erro de API Key!** Verifique suas chaves no arquivo .env")
                    else:
                        st.error(f"❌ **Erro técnico:** {error_msg}")
                
                # Sugestão de solução
                st.info("💡 **Dicas para resolver:**\n- Use OpenAI em vez de Groq\n- Verifique suas API keys\n- Tente um arquivo CSV menor")

def chat_interface():
    """Interface de chat com agentes"""
    st.header("💬 Chat com Agentes Inteligentes")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Carregue um dataset primeiro!")
        return
    
    # Mostrar informações do dataset atual
    if st.session_state.get('current_dataset_info'):
        dataset_info = st.session_state.current_dataset_info
        dataset_source = dataset_info.get('source', '')
        dataset_display = dataset_source
        
        # Encurtar URL se muito longa
        if len(dataset_display) > 60:
            dataset_display = dataset_display[:57] + "..."
            
        st.markdown(f"""
        <div class="dataset-info">
        <strong>📊 Dataset Ativo:</strong> {dataset_info.get('name', 'Desconhecido')}<br>
        <strong>📍 Fonte:</strong> {dataset_display}<br>
        <strong>⏰ Carregado:</strong> {dataset_info.get('loaded_at', 'N/A')[:16].replace('T', ' ')}
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar histórico de conversas
    if st.session_state.chat_history:
        st.subheader("📃 Histórico de Conversas")
        
        for i, chat in enumerate(st.session_state.chat_history):
            if chat['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message">
                <strong>👤 Você:</strong> {chat["message"]}
                </div>
                """, unsafe_allow_html=True)
                
            elif chat['type'] == 'agent':
                st.markdown(f"""
                <div class="agent-response">
                <strong>🤖 Agente:</strong><br>
                {chat["response"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Mostrar gráficos se houver
                if 'image_path' in chat and chat.get('image_path'):
                    if os.path.exists(chat['image_path']):
                        try:
                            image = Image.open(chat['image_path'])
                            st.image(image, caption="Visualização gerada", use_column_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ Erro ao exibir gráfico: {str(e)}")
    
    # Nova pergunta
    st.subheader("❓ Faça sua Pergunta")
    
    # Exemplos em expander
    with st.expander("💡 Exemplos de Perguntas Úteis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Informações Básicas:**
            - Qual arquivo CSV estamos analisando?
            - Quantas linhas e colunas tem o dataset?
            - Quais são os tipos de dados?
            - Existem valores nulos ou faltantes?
            - Mostre as primeiras linhas dos dados
            """)
            
        with col2:
            st.markdown("""
            **🔍 Análises Avançadas:**
            - Mostre estatísticas descritivas completas
            - Existem outliers nos dados?
            - Como as variáveis se relacionam entre si?
            - Crie um gráfico de correlação
            - Quais são suas conclusões sobre os dados?
            """)
    
    # Campo de pergunta
    user_question = st.text_area(
        "Digite sua pergunta sobre os dados:",
        height=100,
        placeholder="Ex: Qual arquivo CSV estamos analisando?",
        key="user_question_input"
    )
    
    # SEÇÃO DE BOTÕES - Layout garantido
    st.markdown("---")
    st.markdown("### 🎛️ Ações Disponíveis")
    
    # Criar 4 colunas para os botões
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        btn_enviar = st.button(
            label="🤖 Enviar",
            key="btn_enviar_question",
            help="Enviar pergunta para análise pelos agentes",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        btn_conclusoes = st.button(
            label="📋 Conclusões", 
            key="btn_get_conclusions",
            help="Obter conclusões consolidadas de todas as análises",
            use_container_width=True
        )
    
    with col3:
        btn_nova_sessao = st.button(
            label="🔄 Nova Sessão",
            key="btn_restart_session", 
            help="Limpar histórico e carregar novo dataset",
            use_container_width=True
        )
    
    with col4:
        btn_finalizar = st.button(
            label="✅ Finalizar",
            key="btn_finalize_session",
            help="Encerrar sessão atual",
            use_container_width=True,
            type="secondary"
        )
    
    # Processar ações dos botões
    if btn_enviar:
        if user_question and user_question.strip():
            process_user_question(user_question.strip())
        else:
            st.warning("⚠️ Digite uma pergunta primeiro!")
    
    if btn_conclusoes:
        get_final_conclusions()
    
    if btn_nova_sessao:
        restart_session_to_upload()
    
    if btn_finalizar:
        finalize_session()

def process_user_question(question: str):
    """Processa pergunta do usuário com contexto do dataset"""
    with st.spinner(f"🤖 Agentes analisando: {question[:50]}..."):
        try:
            # Adicionar contexto do dataset à pergunta para melhor resposta
            dataset_info = st.session_state.get('current_dataset_info', {})
            
            # Mostrar progresso
            progress = st.progress(0)
            status = st.empty()
            
            status.text("🧠 Processando pergunta...")
            progress.progress(30)
            
            status.text("🤖 Agentes colaborando...")
            progress.progress(60)
            
            # Executar análise
            result = st.session_state.eda_system.analyze_question(question)
            
            status.text("📝 Preparando resposta...")
            progress.progress(90)
            
            # Limpar progresso
            progress.progress(100)
            progress.empty()
            status.empty()
            
            # Adicionar ao histórico
            st.session_state.chat_history.append({
                'type': 'user',
                'message': question,
                'timestamp': datetime.now().isoformat()
            })
            
            st.session_state.chat_history.append({
                'type': 'agent', 
                'response': result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Rerun para mostrar nova conversa
            st.rerun()
            
        except Exception as e:
            error_msg = str(e)
            if not handle_rate_limit_error(error_msg, st.session_state.get('current_config', '')):
                st.error(f"❌ **Erro ao processar pergunta:** {error_msg}")

def get_final_conclusions():
    """Obter conclusões finais consolidadas"""
    with st.spinner("🤖 Gerando conclusões consolidadas..."):
        try:
            progress = st.progress(0)
            status = st.empty()
            
            status.text("📊 Revisando todas as análises...")
            progress.progress(40)
            
            status.text("🧠 Consolidando insights...")
            progress.progress(70)
            
            conclusions = st.session_state.eda_system.get_conclusions()
            
            status.text("📋 Preparando relatório final...")
            progress.progress(100)
            
            # Limpar progresso
            progress.empty()
            status.empty()
            
            # Adicionar conclusões ao histórico
            dataset_info = st.session_state.get('current_dataset_info', {})
            dataset_name = dataset_info.get('name', 'Dataset')
            
            st.session_state.chat_history.append({
                'type': 'agent',
                'response': f"**📋 CONCLUSÕES FINAIS - {dataset_name}:**\n\n{conclusions}",
                'timestamp': datetime.now().isoformat(),
                'is_conclusion': True
            })
            
            st.rerun()
            
        except Exception as e:
            error_msg = str(e)
            if not handle_rate_limit_error(error_msg, st.session_state.get('current_config', '')):
                st.error(f"❌ **Erro ao gerar conclusões:** {error_msg}")

def finalize_session():
    """Finalizar sessão com resumo"""
    st.session_state.session_finalized = True
    
    st.markdown("""
    <div class="success-box">
    <h3>✅ Sessão Finalizada com Sucesso!</h3>
    <p><strong>Obrigado por usar o EDA Agente Inteligente!</strong></p>
    <p>Sua análise exploratória foi concluída.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estatísticas da sessão
    if st.session_state.chat_history:
        total_questions = len([c for c in st.session_state.chat_history if c['type'] == 'user'])
        dataset_info = st.session_state.get('current_dataset_info', {})
        dataset_name = dataset_info.get('name', 'Dataset')
        
        st.markdown(f"""
        <div class="dataset-info">
        <strong>📊 Resumo da Sessão:</strong><br>
        • <strong>Dataset analisado:</strong> {dataset_name}<br>
        • <strong>Perguntas respondidas:</strong> {total_questions}<br>
        • <strong>Análises realizadas:</strong> {len(st.session_state.chat_history)} interações<br>
        • <strong>Tempo de sessão:</strong> {datetime.now().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
    
    # Opções finais
    st.markdown("### 📋 Opções Finais")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Baixar Histórico", key="download_history_final", use_container_width=True):
            download_history()
    
    with col2:
        if st.button("🚀 Nova Sessão", key="new_session_from_final", use_container_width=True):
            restart_session_to_upload()

def download_history():
    """Download do histórico da sessão"""
    if st.session_state.chat_history:
        history_data = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "system_config": st.session_state.get('current_config', 'unknown'),
                "dataset_info": st.session_state.get('current_dataset_info', {})
            },
            "conversation": st.session_state.chat_history,
            "total_interactions": len(st.session_state.chat_history),
            "questions_asked": len([c for c in st.session_state.chat_history if c['type'] == 'user'])
        }
        
        json_str = json.dumps(history_data, indent=2, ensure_ascii=False)
        
        # Nome do arquivo baseado no dataset
        dataset_name = st.session_state.get('current_dataset_info', {}).get('name', 'dataset')
        if dataset_name.endswith('.csv'):
            dataset_name = dataset_name[:-4]
        
        filename = f"eda_analysis_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        st.download_button(
            label="📥 Download Histórico (JSON)",
            data=json_str,
            file_name=filename,
            mime="application/json",
            help="Baixar histórico completo da sessão de análise",
            key="download_btn"
        )

def restart_session_to_upload():
    """Reiniciar sessão - VOLTA PARA PÁGINA DE UPLOAD"""
    # Limpar APENAS dados da sessão, mantém sistema inicializado
    session_keys_to_clear = [
        'dataset_loaded', 
        'chat_history', 
        'current_dataset_info', 
        'dataset_source', 
        'session_finalized'
    ]
    
    for key in session_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    clean_temp_files()
    st.success("🔄 **Nova sessão iniciada!** Carregue um novo dataset para análise.")
    st.rerun()

def main():
    """Função principal da aplicação"""
    # Cabeçalho principal
    st.markdown("""
    <div class="main-header">
        <h1>🤖 EDA Agente Inteligente</h1>
        <p>Análise Exploratória de Dados com Agentes CrewAI</p>
        <small>Powered by Groq & OpenAI</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar todos os estados da sessão
    initialize_session_state()
    
    # Se sessão foi finalizada, mostrar apenas página final
    if st.session_state.get('session_finalized'):
        finalize_session()
        return
    
    # Configurar sidebar e obter configurações
    llm_provider, model_name, max_tokens = setup_sidebar()
    
    # Verificar se API keys estão configuradas
    issues = Config.validate_keys()
    if issues:
        st.error("⚠️ **API Keys não configuradas!**")
        st.markdown("""
        **Para usar o sistema:**
        1. **Edite o arquivo `.env`** na pasta do projeto
        2. **Adicione suas chaves API:**
           ```
           GROQ_API_KEY=gsk_sua_chave_groq_aqui
           OPENAI_API_KEY=sk_sua_chave_openai_aqui
           ```
        3. **Reinicie a aplicação** (Ctrl+C e execute novamente)
        
        **Como obter as chaves:**
        - **Groq**: https://console.groq.com → API Keys
        - **OpenAI**: https://platform.openai.com → API Keys
        """)
        st.stop()
    
    # Inicializar sistema EDA
    system_ready = initialize_eda_system(llm_provider, model_name, max_tokens)
    
    if not system_ready:
        st.error("❌ **Sistema não pôde ser inicializado.** Verifique as configurações na sidebar.")
        st.info("💡 **Dica:** Se usar Groq, tente OpenAI ou reduza o limite de tokens.")
        return
    
    # Interface principal baseada no estado
    if not st.session_state.dataset_loaded:
        # Página de carregamento de dataset
        dataset_source = load_dataset_section()
        
        # CORREÇÃO: BOTÃO FINALIZAR SEMPRE VISÍVEL NA PÁGINA INICIAL
        # Mesmo quando não há dataset selecionado, mostrar botão Finalizar
        if not dataset_source:
            st.markdown("---")
            st.markdown("### ✅ Opções Gerais")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button(
                    "✅ Finalizar", 
                    key="finalizar_inicial",
                    use_container_width=True,
                    help="Encerrar aplicação"
                ):
                    finalize_session()
        
        # Se há dataset selecionado, mostrar botão de processamento
        if dataset_source:
            process_dataset_loading(dataset_source)
    else:
        # Interface de chat com dataset carregado
        chat_interface()
        
        # Opção na sidebar para carregar novo dataset
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔄 Opções da Sessão")
        if st.sidebar.button("📁 Carregar Novo Dataset", use_container_width=True, key="sidebar_new_dataset"):
            restart_session_to_upload()
        
        if st.sidebar.button("✅ Finalizar Sessão", use_container_width=True, key="sidebar_finalize"):
            finalize_session()
    
    # Rodapé informativo
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🤖 <strong>EDA Agente Inteligente</strong> | 
        Powered by <strong>CrewAI</strong> • <strong>Streamlit</strong> • <strong>Groq</strong>/<strong>OpenAI</strong><br>
        <small>Use o botão "✅ Finalizar" para encerrar sessões • Versão Sincronizada</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Limpeza automática de arquivos temporários
    clean_temp_files()

if __name__ == "__main__":
    main()