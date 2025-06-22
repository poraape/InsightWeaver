# app_final.py
# Versão final com código modularizado e feedback visual aprimorado.

import streamlit as st

# DDR-EXPANSION (Clarity & Loosely Coupled): Importa a lógica de negócio do módulo de agentes.
from agents import agent_unzip_and_read, agent_query_llm, agent_present_results

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
    # Esta linha não precisa de mudança, pois a configuração do genai é usada pelos agentes importados.
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
            # DDR-EXPANSION (Clarity): Usando st.status para feedback detalhado.
            with st.status("Processando seu arquivo...", expanded=True) as status:
                st.write("Descompactando arquivos...")
                dfs, manifesto = agent_unzip_and_read(uploaded_file)
                
                if dfs:
                    st.session_state.dataframes = dfs
                    st.session_state.manifesto = manifesto
                    st.session_state.data_loaded = True
                    st.session_state.messages = []
                    status.update(label="Análise concluída! Pode começar.", state="complete", expanded=False)
                else:
                    st.error("Nenhum arquivo CSV válido encontrado.")
                    status.update(label="Falha na análise.", state="error", expanded=True)
                    st.session_state.data_loaded = False
        else:
            st.warning("Por favor, faça o upload de um arquivo .zip primeiro.")

# Exibição do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message and message["result"] is not None:
            # A apresentação do resultado agora é parte da mensagem do assistente
            agent_present_results(message["result"], message["original_question"])

# Input do usuário
if prompt := st.chat_input("Qual a sua pergunta sobre os dados?"):
    if not st.session_state.data_loaded:
        st.warning("Por favor, carregue e analise um arquivo na barra lateral primeiro.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # DDR-EXPANSION (Clarity): Usando st.status para o processo de consulta.
            with st.status("Pensando...", expanded=True) as status:
                st.write("Traduzindo sua pergunta para uma consulta...")
                result, code = agent_query_llm(prompt, st.session_state.manifesto, st.session_state.dataframes)
                
                if isinstance(result, str) and result.startswith("Erro"):
                    response_content = f"Desculpe, encontrei um erro: {result}"
                    st.error(response_content)
                    status.update(label="Erro na consulta.", state="error")
                    st.session_state.messages.append({"role": "assistant", "content": response_content, "result": None})
                else:
                    st.write("Apresentando os resultados...")
                    response_content = f"Para responder à sua pergunta, executei a seguinte consulta e obtive estes resultados:"
                    st.markdown(response_content)
                    status.update(label="Resposta gerada!", state="complete", expanded=False)
                    
                    # Adiciona a mensagem completa ao histórico para ser renderizada
                    st.session_state.messages.append({
                        "role": "assistant", "content": response_content,
                        "result": result, "original_question": prompt
                    })
                    # Força a re-execução para exibir a nova mensagem imediatamente
                    st.rerun()
