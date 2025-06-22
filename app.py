# app_v3_final.py
# Versão final com fluxo de agentes completo: Leitura -> Sanitização -> Consulta

import streamlit as st

# DDR-EXPANSION: Importa o novo agente de sanitização
from agents import (
    agent_unzip_and_read, 
    agent_sanitize_and_enrich, 
    agent_query_llm, 
    agent_present_results
)

# --- 1. Configuração da Página e Estilo ---
st.set_page_config(
    page_title="Insight Weaver",
    page_icon=" weaver.png",
    layout="centered"
)
st.markdown("""
<style>
    .stApp { background-color: #f0f2f5; }
    h1 { color: #1d1d1f; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- 2. Configuração da API (Segredo do Servidor) ---
try:
    import google.generativeai as genai
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, AttributeError):
    st.error("A chave de API do Gemini não foi encontrada. Desenvolvedor, configure o segredo 'GEMINI_API_KEY' nas configurações do Streamlit Cloud.")
    st.stop()

# --- 3. Lógica Principal da Interface ---

st.title("`Insight Weaver`")
st.markdown("Faça perguntas em linguagem natural sobre seus arquivos CSV.")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

with st.sidebar:
    st.header("Análise de Dados")
    uploaded_file = st.file_uploader("1. Faça upload de um arquivo .zip", type="zip")

    if st.button("2. Analisar Dados", use_container_width=True):
        if uploaded_file:
            with st.status("Processando seus dados...", expanded=True) as status:
                # DDR-EXPANSION: Novo fluxo de orquestração com 3 etapas
                
                status.write("Etapa 1: Lendo arquivos...")
                raw_dfs = agent_unzip_and_read(uploaded_file)
                
                if raw_dfs:
                    status.write("Etapa 2: Limpando e otimizando os dados...")
                    sanitized_dfs, manifesto = agent_sanitize_and_enrich(raw_dfs)
                    
                    st.session_state.dataframes = sanitized_dfs
                    st.session_state.manifesto = manifesto
                    st.session_state.data_loaded = True
                    st.session_state.messages = []
                    status.update(label="Análise concluída! Pode começar.", state="complete", expanded=False)
                else:
                    st.error("Nenhum arquivo CSV válido encontrado.")
                    status.update(label="Falha na leitura.", state="error", expanded=True)
                    st.session_state.data_loaded = False
        else:
            st.warning("Por favor, faça o upload de um arquivo .zip primeiro.")

# O resto do código para exibição do chat e input do usuário permanece o mesmo
# ... (cole o resto do código de `app_final.py` aqui)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message and message["result"] is not None:
            agent_present_results(message["result"], message["original_question"])

if prompt := st.chat_input("Qual a sua pergunta sobre os dados?"):
    if not st.session_state.data_loaded:
        st.warning("Por favor, carregue e analise um arquivo na barra lateral primeiro.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Pensando...", expanded=True) as status:
                status.write("Traduzindo sua pergunta para uma consulta...")
                result, code = agent_query_llm(prompt, st.session_state.manifesto, st.session_state.dataframes)
                
                if isinstance(result, str) and result.startswith("Erro"):
                    response_content = f"Desculpe, encontrei um erro: {result}"
                    st.error(response_content)
                    status.update(label="Erro na consulta.", state="error")
                    st.session_state.messages.append({"role": "assistant", "content": response_content, "result": None})
                else:
                    status.write("Apresentando os resultados...")
                    response_content = f"Para responder à sua pergunta, executei a seguinte consulta e obtive estes resultados:"
                    st.markdown(response_content)
                    status.update(label="Resposta gerada!", state="complete", expanded=False)
                    
                    st.session_state.messages.append({
                        "role": "assistant", "content": response_content,
                        "result": result, "original_question": prompt
                    })
                    st.rerun()
