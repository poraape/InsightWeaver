# app_final_proactive.py
# Versão final com sugestões de perguntas inteligentes.

import streamlit as st

from agents import (
    agent_unzip_and_read, 
    agent_sanitize_and_enrich, 
    agent_suggest_questions, # Importa o novo agente
    agent_query_llm, 
    agent_present_results
)

# --- Configuração da Página e API ---
st.set_page_config(page_title="Insight Weaver", page_icon=" weaver.png", layout="centered")
st.markdown("""<style>.stApp { background-color: #f0f2f5; } h1 { color: #1d1d1f; font-weight: 600; }</style>""", unsafe_allow_html=True)
try:
    import google.generativeai as genai
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, AttributeError):
    st.error("Chave de API do Gemini não configurada. Configure o segredo 'GEMINI_API_KEY'.")
    st.stop()

# --- Lógica Principal da Interface ---

st.title("`Insight Weaver`")
st.markdown("Faça perguntas em linguagem natural sobre seus arquivos CSV.")

# Inicialização do estado da sessão
if 'messages' not in st.session_state: st.session_state.messages = []
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'suggested_questions' not in st.session_state: st.session_state.suggested_questions = []

# Função para processar uma pergunta (evita duplicação de código)
def process_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status("Pensando...", expanded=True) as status:
            status.write("Traduzindo sua pergunta...")
            result, code = agent_query_llm(question, st.session_state.manifesto, st.session_state.dataframes)
            if isinstance(result, str) and result.startswith("Falha"):
                response_content = f"Desculpe, encontrei um erro: {result}"
                st.error(response_content)
                status.update(label="Erro na consulta.", state="error")
                st.session_state.messages.append({"role": "assistant", "content": response_content, "result": None})
            else:
                status.write("Apresentando os resultados...")
                response_content = f"Para responder, executei a seguinte consulta:"
                st.markdown(response_content)
                status.update(label="Resposta gerada!", state="complete", expanded=False)
                st.session_state.messages.append({
                    "role": "assistant", "content": response_content,
                    "result": result, "original_question": question
                })
                st.rerun()

# Sidebar
with st.sidebar:
    st.header("Análise de Dados")
    uploaded_file = st.file_uploader("1. Faça upload de um arquivo .zip", type="zip")
    if st.button("2. Analisar Dados", use_container_width=True):
        if uploaded_file:
            with st.status("Processando seus dados...", expanded=True) as status:
                status.write("Etapa 1: Lendo arquivos...")
                raw_dfs = agent_unzip_and_read(uploaded_file)
                if raw_dfs:
                    status.write("Etapa 2: Limpando e otimizando...")
                    sanitized_dfs, manifesto = agent_sanitize_and_enrich(raw_dfs)
                    st.session_state.dataframes = sanitized_dfs
                    st.session_state.manifesto = manifesto
                    
                    status.write("Etapa 3: Gerando sugestões inteligentes...")
                    st.session_state.suggested_questions = agent_suggest_questions(manifesto)
                    
                    st.session_state.data_loaded = True
                    st.session_state.messages = []
                    status.update(label="Análise concluída!", state="complete", expanded=False)
                else:
                    st.error("Nenhum arquivo CSV válido encontrado.")
                    status.update(label="Falha na leitura.", state="error")
        else:
            st.warning("Por favor, faça o upload de um arquivo .zip primeiro.")

# Exibição do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message and message["result"] is not None:
            agent_present_results(message["result"], message["original_question"])

# DDR-EXPANSION: Exibição das perguntas sugeridas
if st.session_state.data_loaded and not st.session_state.messages:
    st.markdown("### Que tal começar com uma destas perguntas?")
    cols = st.columns(len(st.session_state.suggested_questions))
    for i, question in enumerate(st.session_state.suggested_questions):
        with cols[i]:
            if st.button(question, use_container_width=True):
                process_question(question)
                st.rerun()

# Input do usuário
if prompt := st.chat_input("Qual a sua pergunta sobre os dados?"):
    if not st.session_state.data_loaded:
        st.warning("Por favor, carregue e analise um arquivo na barra lateral primeiro.")
    else:
        process_question(prompt)
