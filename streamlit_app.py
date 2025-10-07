import streamlit as st
import pandas as pd
import os
import tempfile
import json
import numpy as np
from datetime import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
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

# Dark Theme CSS
st.markdown("""
<style>
    body {
        color: #f0f2f6;
        background-color: #0e1117;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #3a3a3a 0%, #2a2a2a 100%);
        color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #4a4a4a;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #7b68ee;
        background-color: #262730;
    }
    .agent-response {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        background-color: #1e2a24;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #3b310a;
        border: 1px solid #a78d4f;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #1c3344;
        border: 1px solid #3d85c6;
        margin: 1rem 0;
    }
    .dataset-info {
        padding: 1rem;
        border-radius: 10px;
        background-color: #2a2a3a;
        border: 1px solid #4a4a5a;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
        border-radius: 5px;
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
    """Função para tratar rate limits"""
    if "rate limit" in error_msg.lower() or "rate_limit_exceeded" in error_msg:
        st.error("Rate Limit Atingido! Reduza tokens na sidebar ou mude para OpenAI.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reduzir Tokens", key="reduce_tokens"):
                if provider == "groq":
                    st.session_state.tokens_slider = 200
                    st.rerun()
        with col2:
            if st.button("Mudar OpenAI", key="switch_openai"):
                st.session_state.llm_provider_select = "openai"
                st.rerun()
        
        return True
    return False


# Sidebar geral para qualquer dataset
def add_visualization_sidebar():
    """Sidebar com opções gerais de visualização - VERSÃO CORRIGIDA"""
    if st.session_state.get('dataset_loaded') and st.session_state.get('eda_system'):
        
        eda_system = st.session_state.eda_system

        if hasattr(eda_system, 'current_dataset') and eda_system.current_dataset is not None:
            data = eda_system.current_dataset

            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Visualizações Rápidas")

            # Botão principal para gerar todas as visualizações
            if st.sidebar.button("🎯 Gerar Todas as Visualizações", key="viz_all_sidebar"):
                # Simular input do usuário para processar visualizações
                question = "visualizações completas do dataset"

                # Processar sem rerun automático
                process_user_question(question)

                # Mostrar mensagem de sucesso
                st.sidebar.success("✅ Visualizações geradas!")

            # Botões para gráficos específicos
            st.sidebar.markdown("**Gráficos Específicos:**")

            if st.sidebar.button("📈 Matriz Correlação", key="viz_corr"):
                process_user_question("gera matriz de correlação")
                st.sidebar.success("✅ Matriz gerada!")

            if st.sidebar.button("📊 Distribuições", key="viz_dist"):
                process_user_question("mostra distribuições das variáveis")
                st.sidebar.success("✅ Distribuições geradas!")

            if st.sidebar.button("🎯 Scatter Plots", key="viz_scatter"):
                process_user_question("cria scatter plots entre variáveis")
                st.sidebar.success("✅ Scatter plots gerados!")

            # Informações do dataset
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📋 Info do Dataset")

            # Métricas básicas
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Linhas", data.shape[0])
            with col2:
                st.metric("Colunas", data.shape[1])

            # Tipos de colunas
            numeric_cols = data.select_dtypes(include=['number']).columns
            categorical_cols = data.select_dtypes(include=['object']).columns

            st.sidebar.markdown(f"**Numéricas:** {len(numeric_cols)}")
            st.sidebar.markdown(f"**Categóricas:** {len(categorical_cols)}")

            # Mostrar nomes das colunas principais
            st.sidebar.markdown("**Principais Colunas:**")

            if len(data.columns) <= 8:
                for col in data.columns:
                    # Encurtar nomes longos
                    col_display = col if len(col) <= 20 else col[:17] + "..."
                    st.sidebar.markdown(f"• {col_display}")
            else:
                for col in data.columns[:6]:
                    col_display = col if len(col) <= 20 else col[:17] + "..."
                    st.sidebar.markdown(f"• {col_display}")
                st.sidebar.markdown(f"... e mais {len(data.columns)-6}")

            # Status de qualidade dos dados
            st.sidebar.markdown("---")
            st.sidebar.markdown("### ⚡ Qualidade dos Dados")

            # Verificar valores nulos
            null_count = data.isnull().sum().sum()
            if null_count > 0:
                st.sidebar.warning(f"⚠️ {null_count} valores nulos")
            else:
                st.sidebar.success("✅ Sem valores nulos")

            # Verificar duplicatas
            dup_count = data.duplicated().sum()
            if dup_count > 0:
                st.sidebar.warning(f"⚠️ {dup_count} linhas duplicadas")
            else:
                st.sidebar.success("✅ Sem duplicatas")

            # Tamanho do dataset
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            st.sidebar.info(f"💾 Tamanho: {memory_usage:.1f} MB")

def setup_sidebar():
    """Configura a barra lateral"""
    st.sidebar.header("Configurações do Sistema")

    # API Key Inputs
    st.sidebar.subheader("API Keys")
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Insira sua chave API da OpenAI.",
        key="openai_api_key_input"
    )
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Insira sua chave API da Groq.",
        key="groq_api_key_input"
    )

    # Update environment variables if keys are provided
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
    if groq_api_key:
        os.environ['GROQ_API_KEY'] = groq_api_key
    
    llm_provider = st.sidebar.selectbox(
        "Provedor LLM:",
        ["openai", "groq"],
        help="OpenAI é mais estável. Groq é rápido mas tem rate limits",
        key="llm_provider_select"
    )
    
    model_options = {
        "groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b"
        ],
        "openai": [
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ]
    }
    
    model_name = st.sidebar.selectbox(
        "Modelo:",
        model_options[llm_provider],
        index=0,
        key="model_select"
    )
    
    if llm_provider == "groq":
        st.sidebar.warning("Groq: Rate limits severos")
        
        max_tokens = st.sidebar.slider(
            "Limite Tokens:",
            min_value=100,
            max_value=600,
            value=300,
            step=50,
            key="tokens_slider"
        )
        
        st.session_state.max_tokens_groq = max_tokens
    else:
        st.sidebar.success("OpenAI: Mais estável")
        max_tokens = 1500
    
    # Status das API Keys
    st.sidebar.subheader("Status das Chaves")
    # This check is now just for display purposes
    if not os.environ.get('OPENAI_API_KEY'):
        st.sidebar.warning("Chave OpenAI não fornecida.")
    else:
        st.sidebar.success("Chave OpenAI configurada.")

    if not os.environ.get('GROQ_API_KEY'):
        st.sidebar.warning("Chave Groq não fornecida.")
    else:
        st.sidebar.success("Chave Groq configurada.")
    
    return llm_provider, model_name, max_tokens

def load_dataset_section():
    """Seção de carregamento de dataset"""
    st.header("Carregamento de Dataset CSV")
    
    tab1, tab2, tab3 = st.tabs(["Upload Local", "URL Direta", "Exemplos"])
    
    dataset_source = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV:",
            type=['csv'],
            help="Upload de qualquer arquivo CSV do seu computador",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            temp_dir = Config.UPLOAD_DIR
            ensure_directories()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            dataset_source = temp_path
            st.text_input(
                "Arquivo selecionado:", 
                value=uploaded_file.name, 
                disabled=True,
                key="file_display"
            )
            st.success(f"Arquivo salvo: {uploaded_file.name}")
    
    with tab2:
        csv_url = st.text_input(
            "URL do arquivo CSV:",
            placeholder="https://exemplo.com/dados.csv",
            help="Cole a URL direta de qualquer arquivo CSV público",
            key="csv_url_input"
        )
        
        if csv_url and csv_url.strip():
            url_clean = csv_url.strip()
            if url_clean.startswith(('http://', 'https://')):
                dataset_source = url_clean
                url_filename = url_clean.split('/')[-1]
                if '?' in url_filename:
                    url_filename = url_filename.split('?')[0]
                if not url_filename:
                    url_filename = "arquivo.csv"
                    
                st.text_input(
                    "Arquivo da URL:", 
                    value=url_filename, 
                    disabled=True,
                    key="url_display"
                )
                st.success("URL configurada")
            else:
                st.error("URL inválida! Deve começar com http:// ou https://")
    
    with tab3:
        st.markdown("**Datasets de Exemplo (apenas para demonstração):**")
        
        example_url = st.selectbox(
            "Escolha um exemplo:",
            ["", 
             "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
             "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
             "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"],
            format_func=lambda x: {
                "": "Selecione um exemplo...",
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv": "Iris Dataset (classificação)",
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv": "Tips Dataset (restaurante)",
                "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv": "Titanic Dataset (histórico)"
            }.get(x, x),
            key="example_selector"
        )
        
        if example_url:
            dataset_source = example_url
            example_filename = example_url.split('/')[-1]
            st.text_input(
                "Exemplo selecionado:", 
                value=example_filename, 
                disabled=True,
                key="example_display"
            )
            st.success(f"Exemplo: {example_filename}")
    
    return dataset_source

def initialize_eda_system(llm_provider: str, model_name: str, max_tokens: int):
    """Inicializa sistema EDA"""
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    current_config = f"{llm_provider}-{model_name}-{max_tokens}-{openai_key}-{groq_key}"
    
    if (st.session_state.current_config != current_config or 
        not st.session_state.system_initialized):
        
        with st.spinner(f"Inicializando: {llm_provider.upper()} - {model_name}"):
            try:
                if 'eda_system' in st.session_state:
                    del st.session_state.eda_system
                
                if st.session_state.current_config != current_config:
                    if st.session_state.get('dataset_loaded'):
                        st.session_state.dataset_loaded = False
                        st.session_state.chat_history = []
                        st.warning("Configuração mudou! Recarregue dataset.")
                
                eda_system = EDACrewSystem(llm_provider, model_name, max_tokens)
                
                st.session_state.eda_system = eda_system
                st.session_state.current_config = current_config
                st.session_state.system_initialized = True
                
                st.success(f"Sistema inicializado: {llm_provider.upper()}")
                return True
                
            except Exception as e:
                st.session_state.system_initialized = False
                error_msg = str(e)
                
                if handle_rate_limit_error(error_msg, llm_provider):
                    return False
                elif "api" in error_msg.lower() and "key" in error_msg.lower():
                    st.error(f"Erro de API Key - {llm_provider.upper()}")
                else:
                    st.error(f"Erro de inicialização: {error_msg}")
                
                return False
    
    return True

def process_dataset_loading(dataset_source: str):
    """Processa carregamento do dataset"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        load_button = st.button(
            "Processar Carregamento",
            type="primary", 
            use_container_width=True,
            key="load_dataset_btn"
        )
    
    with col2:
        finalizar_load_btn = st.button(
            "Finalizar", 
            key="finalizar_load", 
            use_container_width=True
        )
    
    if finalizar_load_btn:
        finalize_session()
        return
    
    if load_button:
        if not st.session_state.get('system_initialized'):
            st.error("Sistema não inicializado!")
            return
        
        st.session_state.dataset_loaded = False
        st.session_state.chat_history = []
        st.session_state.session_finalized = False
        st.session_state.dataset_source = dataset_source
        
        with st.spinner("Carregando dataset..."):
            try:
                result = st.session_state.eda_system.load_dataset(dataset_source)
                
                has_error = any(keyword in result.lower() for keyword in 
                              ['erro', 'error', 'exception', 'rate limit', 'failed'])
                
                if has_error:
                    st.session_state.dataset_loaded = False
                    if not handle_rate_limit_error(result, st.session_state.get('current_config', '')):
                        st.error("Falha no carregamento")
                        st.markdown(f'<div class="warning-box">{result}</div>', unsafe_allow_html=True)
                else:
                    st.session_state.dataset_loaded = True
                    
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
                    
                    st.markdown('<div class="agent-response">', unsafe_allow_html=True)
                    st.markdown("**Agente Explorador de Dados:**")
                    st.markdown(result)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.session_state.chat_history.append({
                        'type': 'system',
                        'message': f'Dataset carregado: {dataset_name}',
                        'response': result,
                        'dataset_info': st.session_state.current_dataset_info
                    })
                    
                    st.success("Dataset carregado! Agora pode fazer perguntas sobre os dados.")
                    st.balloons()
                    
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button(
                            "Iniciar Análise Interativa", 
                            type="primary", 
                            key="goto_chat",
                            use_container_width=True
                        ):
                            st.rerun()
            
            except Exception as e:
                st.session_state.dataset_loaded = False
                error_msg = str(e)
                
                if not handle_rate_limit_error(error_msg, st.session_state.get('current_config', '')):
                    st.error(f"Erro técnico: {error_msg}")

def chat_interface():
    """Interface de chat com agentes"""
    st.header("Chat com Agentes Inteligentes")
    
    if not st.session_state.dataset_loaded:
        st.warning("Carregue um dataset primeiro!")
        return
    
    # Mostrar informações do dataset atual
    if st.session_state.get('current_dataset_info'):
        dataset_info = st.session_state.current_dataset_info
        dataset_source = dataset_info.get('source', '')
        dataset_display = dataset_source
        
        if len(dataset_display) > 60:
            dataset_display = dataset_display[:57] + "..."
            
        st.markdown(f"""
        <div class="dataset-info">
        <strong>Dataset Ativo:</strong> {dataset_info.get('name', 'Desconhecido')}<br>
        <strong>Fonte:</strong> {dataset_display}<br>
        <strong>Carregado:</strong> {dataset_info.get('loaded_at', 'N/A')[:16].replace('T', ' ')}
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar histórico de conversas
    if st.session_state.chat_history:
        st.subheader("Histórico de Conversas")
        
        for i, chat in enumerate(st.session_state.chat_history):
            if chat['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message">
                <strong>Você:</strong> {chat["message"]}
                </div>
                """, unsafe_allow_html=True)
                
            elif chat['type'] == 'agent':
                st.markdown(f"""
                <div class="agent-response">
                <strong>Agente:</strong><br>
                {chat["response"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Nova pergunta
    st.subheader("Faça sua Pergunta")
    
    # Exemplos gerais (não específicos do Titanic)
    with st.expander("Exemplos de Perguntas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Informações Básicas:**
            - Qual arquivo CSV estamos analisando?
            - Quantas linhas e colunas tem o dataset?
            - Quais são os tipos de dados?
            - Existem valores nulos ou faltantes?
            - Mostre as primeiras linhas dos dados
            """)
            
        with col2:
            st.markdown("""
            **Análises com Gráficos:**
            - Gera gráfico ilustrando tema do arquivo
            - Cria matriz de correlação entre variáveis
            - Mostra distribuição das variáveis numéricas
            - Faz análise das variáveis categóricas
            - Visualiza todas as relações nos dados
            """)
    
    # Campo de pergunta
    user_question = st.text_area(
        "Digite sua pergunta sobre os dados:",
        height=100,
        placeholder="Ex: Gera gráfico ilustrando tema do arquivo",
        key="user_question_input"
    )
    
    # Botões de ação
    st.markdown("---")
    st.markdown("### Ações Disponíveis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        btn_enviar = st.button(
            label="Enviar",
            key="btn_enviar_question",
            help="Enviar pergunta para análise",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        btn_conclusoes = st.button(
            label="Conclusões", 
            key="btn_get_conclusions",
            help="Obter conclusões consolidadas",
            use_container_width=True
        )
    
    with col3:
        btn_nova_sessao = st.button(
            label="Nova Sessão",
            key="btn_restart_session", 
            help="Carregar novo dataset",
            use_container_width=True
        )
    
    with col4:
        btn_finalizar = st.button(
            label="Finalizar",
            key="btn_finalize_session",
            help="Encerrar sessão atual",
            use_container_width=True,
            type="secondary"
        )
    
    # Processar ações
    if btn_enviar:
        if user_question and user_question.strip():
            process_user_question(user_question.strip())
        else:
            st.warning("Digite uma pergunta primeiro!")
    
    if btn_conclusoes:
        get_final_conclusions()
    
    if btn_nova_sessao:
        restart_session_to_upload()
    
    if btn_finalizar:
        finalize_session()

def process_user_question(question: str):
    """Versão corrigida sem modificação de widget"""

    # Detectar solicitações de visualização
    viz_keywords = ['gráfico', 'grafico', 'chart', 'plot', 'visualiza',
                   'correlação', 'correlacao', 'distribuição', 'distribuicao',
                   'heatmap', 'histograma', 'mostra', 'gera', 'cria', 'ilustra',
                   'matriz', 'scatter', 'todas as visualizações', 'visualizações completas']

    is_viz_request = any(keyword in question.lower() for keyword in viz_keywords)

    # Adicionar pergunta ao histórico ANTES do processamento
    st.session_state.chat_history.append({
        'type': 'user',
        'message': question,
        'timestamp': datetime.now().isoformat()
    })

    if is_viz_request:
        # PROCESSAMENTO DE VISUALIZAÇÕES
        st.markdown("## 📊 GRÁFICOS GERADOS")

        try:
            # Acessar dados diretamente
            eda_system = st.session_state.eda_system
            data = eda_system.current_dataset
            dataset_name = eda_system.dataset_info.get('name', 'Dataset')

            st.success(f"✅ Gerando gráficos para: {dataset_name}")

            # Preparar contadores para keys únicos
            chart_counter = len(st.session_state.chat_history)

            # Identificar tipos de colunas
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

            charts_generated = []

            # 1. INFORMAÇÕES GERAIS DO DATASET
            st.markdown("### 📋 Resumo do Dataset")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Linhas", data.shape[0])
            with col2:
                st.metric("Colunas", data.shape[1])
            with col3:
                st.metric("Numéricas", len(numeric_cols))
            with col4:
                st.metric("Categóricas", len(categorical_cols))

            # 2. MATRIZ DE CORRELAÇÃO
            if len(numeric_cols) >= 2:
                st.markdown("### 🔗 Matriz de Correlação")

                corr_matrix = data[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title="Matriz de Correlação entre Variáveis Numéricas",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                fig_corr.update_layout(width=800, height=600)

                # USAR CONTAINER ÚNICO PARA EVITAR CONFLITOS
                st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_matrix_{chart_counter}")
                charts_generated.append("Matriz de Correlação")

                # Análise das correlações mais fortes
                corr_abs = corr_matrix.abs()
                np.fill_diagonal(corr_abs.values, 0)
                if not corr_abs.empty and corr_abs.max().max() > 0:
                    max_corr = corr_abs.max().max()
                    max_corr_idx = corr_abs.stack().idxmax()
                    actual_corr = corr_matrix.loc[max_corr_idx[0], max_corr_idx[1]]
                    st.info(f"Correlação mais forte: {max_corr_idx[0]} ↔ {max_corr_idx[1]} ({actual_corr:.3f})")

            # 3. DISTRIBUIÇÕES DAS VARIÁVEIS NUMÉRICAS
            if len(numeric_cols) > 0:
                st.markdown("### 📊 Distribuições das Variáveis Numéricas")

                # Mostrar até 4 distribuições em 2 colunas
                num_plots = min(len(numeric_cols), 4)

                for i, col in enumerate(numeric_cols[:num_plots]):
                    st.markdown(f"**Distribuição de {col}:**")

                    # Histograma com marginal box plot
                    fig_hist = px.histogram(
                        data,
                        x=col,
                        title=f"Distribuição: {col}",
                        marginal="box",
                        nbins=min(30, data[col].nunique()) if data[col].nunique() > 2 else 10
                    )

                    st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{col}_{chart_counter}")

                    # Estatísticas básicas
                    stats = data[col].describe()
                    st.caption(f"Média: {stats['mean']:.2f} | Mediana: {stats['50%']:.2f} | Desvio: {stats['std']:.2f}")

                charts_generated.append(f"Distribuições de {num_plots} variáveis")

            # 4. RELAÇÕES ENTRE VARIÁVEIS NUMÉRICAS
            if len(numeric_cols) >= 2:
                st.markdown("### 🎯 Relações entre Variáveis")

                # Scatter plot das duas primeiras variáveis
                x_var = numeric_cols[0]
                y_var = numeric_cols[1]

                # Se há variável categórica, usar para colorir
                if len(categorical_cols) > 0:
                    color_var = categorical_cols[0]
                    unique_cats = data[color_var].nunique()

                    if unique_cats <= 10:  # Máximo 10 categorias
                        fig_scatter = px.scatter(
                            data,
                            x=x_var,
                            y=y_var,
                            color=color_var,
                            title=f"{x_var} vs {y_var} (por {color_var})"
                        )
                    else:
                        fig_scatter = px.scatter(
                            data,
                            x=x_var,
                            y=y_var,
                            title=f"{x_var} vs {y_var}"
                        )
                else:
                    fig_scatter = px.scatter(
                        data,
                        x=x_var,
                        y=y_var,
                        title=f"{x_var} vs {y_var}"
                    )

                st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{chart_counter}")
                charts_generated.append(f"Scatter plot {x_var} vs {y_var}")
            
            # 5. ANÁLISE POR CATEGORIAS (específico para dados como Titanic)
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                st.markdown("### 📈 Análise por Categorias")

                # Procurar colunas de sobrevivência e gênero
                survival_cols = [col for col in categorical_cols if any(word in col.lower()
                                for word in ['survived', 'survival', 'outcome'])]
                gender_cols = [col for col in categorical_cols if any(word in col.lower()
                              for word in ['sex', 'gender'])]

                if survival_cols and gender_cols:
                    survival_col = survival_cols[0]
                    gender_col = gender_cols[0]

                    st.markdown(f"**Análise de {survival_col} por {gender_col}:**")

                    # Gráfico de barras agrupadas
                    fig_survival = px.histogram(
                        data,
                        x=gender_col,
                        color=survival_col,
                        title=f"Distribuição de {survival_col} por {gender_col}",
                        barmode='group'
                    )
                    st.plotly_chart(fig_survival, use_container_width=True, key=f"survival_{chart_counter}")

                    # Tabela cruzada
                    cross_tab = pd.crosstab(data[gender_col], data[survival_col], margins=True)
                    st.dataframe(cross_tab, use_container_width=True)

                    charts_generated.append(f"Análise de {survival_col} por {gender_col}")

                else:
                    # Análise geral por categorias
                    cat_var = categorical_cols[0]
                    num_var = numeric_cols[0]

                    unique_cats = data[cat_var].nunique()
                    if unique_cats <= 15:  # Máximo 15 categorias

                        # Box plot
                        fig_box = px.box(
                            data,
                            x=cat_var,
                            y=num_var,
                            title=f"Distribuição de {num_var} por {cat_var}"
                        )
                        st.plotly_chart(fig_box, use_container_width=True, key=f"box_{chart_counter}")
                        charts_generated.append(f"Box plot {num_var} por {cat_var}")

            # 6. AMOSTRA DOS DADOS
            st.markdown("### 🔍 Amostra dos Dados")
            st.dataframe(data.head(10), use_container_width=True)

            # 7. ESTATÍSTICAS DESCRITIVAS
            if len(numeric_cols) > 0:
                st.markdown("### 📈 Estatísticas Descritivas")
                desc_stats = data[numeric_cols].describe()
                st.dataframe(desc_stats, use_container_width=True)

            # RESPOSTA CONSOLIDADA
            result = f"""✅ **GRÁFICOS EXIBIDOS COM SUCESSO!**

**Dataset analisado:** {dataset_name}
• Dimensões: {data.shape[0]} linhas × {data.shape[1]} colunas
• Variáveis numéricas: {len(numeric_cols)}
• Variáveis categóricas: {len(categorical_cols)}

**Gráficos gerados:**
{chr(10).join([f"• {chart}" for chart in charts_generated])}

**Todos os gráficos são interativos** - você pode fazer zoom, filtrar e explorar os dados.

**Os gráficos estão visíveis ACIMA desta mensagem!** Se não consegue ver, role a tela para cima."""

        except Exception as e:
            st.error(f"Erro ao gerar gráficos: {str(e)}")
            result = f"Erro ao gerar visualizações: {str(e)}"
    
    else:
        # PROCESSAMENTO DE PERGUNTAS NORMAIS
        with st.spinner("🤖 Analisando sua pergunta..."):
            try:
                # Usar o sistema CrewAI para perguntas normais
                result = st.session_state.eda_system.analyze_question(question)

                # Verificar se há erro de rate limit
                if any(keyword in result.lower() for keyword in
                      ['rate limit', 'erro', 'error', 'exception']):

                    current_config = st.session_state.get('current_config', '')
                    if 'groq' in current_config:
                        if not handle_rate_limit_error(result, 'groq'):
                            st.error("Erro na análise da pergunta")
                        return  # Sair sem adicionar ao histórico se há erro
                    else:
                        st.error(f"Erro: {result}")
                        return

            except Exception as e:
                error_msg = str(e)
                current_config = st.session_state.get('current_config', '')

                if 'groq' in current_config and 'rate' in error_msg.lower():
                    if not handle_rate_limit_error(error_msg, 'groq'):
                        st.error(f"Erro técnico: {error_msg}")
                else:
                    st.error(f"Erro ao processar pergunta: {error_msg}")
                return

    # Adicionar resposta ao histórico
    st.session_state.chat_history.append({
        'type': 'agent',
        'response': result,
        'timestamp': datetime.now().isoformat(),
        'has_visualization': is_viz_request
    })

    # MOSTRAR RESPOSTA IMEDIATAMENTE (sem rerun)
    if is_viz_request:
        # Para visualizações, a resposta já foi mostrada acima
        st.markdown("---")
        st.success("✅ Visualizações geradas com sucesso! Veja os gráficos acima.")
    else:
        # Para perguntas normais, mostrar a resposta
        st.markdown(f"""
        <div class="agent-response">
        <strong>🤖 Resposta do Agente:</strong><br>
        {result}
        </div>
        """, unsafe_allow_html=True)

    # REMOÇÃO DA LINHA PROBLEMÁTICA:
    # NÃO podemos modificar st.session_state.user_question_input aqui
    # O Streamlit não permite modificar widgets após instantiação


def get_final_conclusions():
    """Obter conclusões finais consolidadas"""
    with st.spinner("Gerando conclusões consolidadas..."):
        try:
            conclusions = st.session_state.eda_system.get_conclusions()
            
            dataset_info = st.session_state.get('current_dataset_info', {})
            dataset_name = dataset_info.get('name', 'Dataset')
            
            st.session_state.chat_history.append({
                'type': 'agent',
                'response': f"**CONCLUSÕES FINAIS - {dataset_name}:**\n\n{conclusions}",
                'timestamp': datetime.now().isoformat(),
                'is_conclusion': True
            })
            
            st.rerun()
            
        except Exception as e:
            error_msg = str(e)
            if not handle_rate_limit_error(error_msg, st.session_state.get('current_config', '')):
                st.error(f"Erro ao gerar conclusões: {error_msg}")

def finalize_session():
    """Finalizar sessão com resumo"""
    st.session_state.session_finalized = True
    
    st.markdown("""
    <div class="success-box">
    <h3>Sessão Finalizada com Sucesso!</h3>
    <p><strong>Obrigado por usar o EDA Agente Inteligente!</strong></p>
    <p>Sua análise exploratória foi concluída.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        total_questions = len([c for c in st.session_state.chat_history if c['type'] == 'user'])
        dataset_info = st.session_state.get('current_dataset_info', {})
        dataset_name = dataset_info.get('name', 'Dataset')
        
        st.markdown(f"""
        <div class="dataset-info">
        <strong>Resumo da Sessão:</strong><br>
        • <strong>Dataset analisado:</strong> {dataset_name}<br>
        • <strong>Perguntas respondidas:</strong> {total_questions}<br>
        • <strong>Análises realizadas:</strong> {len(st.session_state.chat_history)} interações<br>
        • <strong>Tempo de sessão:</strong> {datetime.now().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Baixar Histórico", key="download_history_final", use_container_width=True):
            download_history()
    
    with col2:
        if st.button("Nova Sessão", key="new_session_from_final", use_container_width=True):
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
        
        dataset_name = st.session_state.get('current_dataset_info', {}).get('name', 'dataset')
        if dataset_name.endswith('.csv'):
            dataset_name = dataset_name[:-4]
        
        filename = f"eda_analysis_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        st.download_button(
            label="Download Histórico (JSON)",
            data=json_str,
            file_name=filename,
            mime="application/json",
            key="download_btn"
        )

def restart_session_to_upload():
    """Reiniciar sessão"""
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
    st.success("Nova sessão iniciada! Carregue um novo dataset.")
    st.rerun()

def main():
    """Função principal da aplicação"""
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔮 EDA Wizard</h1>
        <p>Your personal AI assistant for Exploratory Data Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar estados
    initialize_session_state()
    
    # Se sessão foi finalizada
    if st.session_state.get('session_finalized'):
        finalize_session()
        return
    
    # Configurar sidebar
    llm_provider, model_name, max_tokens = setup_sidebar()
    
    # Sidebar de visualizações
    add_visualization_sidebar()

    # Inicializar sistema
    system_ready = initialize_eda_system(llm_provider, model_name, max_tokens)
    
    if not system_ready:
        st.error("Sistema não pôde ser inicializado.")
        return
    
    # Interface principal
    if not st.session_state.dataset_loaded:
        # Página de carregamento
        dataset_source = load_dataset_section()
        
        if not dataset_source:
            st.markdown("---")
            st.markdown("### Opções Gerais")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("Finalizar", key="finalizar_inicial", use_container_width=True):
                    finalize_session()
        
        if dataset_source:
            process_dataset_loading(dataset_source)
    else:
        # Interface de chat
        chat_interface()
        
        # Opções na sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Opções da Sessão")
        if st.sidebar.button("Carregar Novo Dataset", use_container_width=True, key="sidebar_new_dataset"):
            restart_session_to_upload()
        
        if st.sidebar.button("Finalizar Sessão", use_container_width=True, key="sidebar_finalize"):
            finalize_session()
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <strong>EDA Agente Inteligente</strong> | 
        Powered by <strong>CrewAI</strong> • <strong>Streamlit</strong> • <strong>Groq</strong>/<strong>OpenAI</strong><br>
        <small><strong>Sistema Universal:</strong> Analisa qualquer arquivo CSV com visualizações automáticas</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Limpeza
    clean_temp_files()

if __name__ == "__main__":
    main()
