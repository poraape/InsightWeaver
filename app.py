      
# app_public_version.py
# Insight Weaver: Versão pública sem necessidade de chave de API do usuário.

import streamlit as st
import polars as pl
import google.generativeai as genai
import zipfile
import os
import shutil
from pathlib import Path
import plotly.express as px

# --- Configuração da Página e Estilo (sem alterações) ---
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


# --- DDR-REFACTOR (Abstracted): A chave de API agora é um parâmetro central ---
# É lida uma única vez do ambiente e passada para as funções que a necessitam.
try:
    # Acessa a chave de API armazenada nos segredos do Streamlit
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    # Mensagem de erro para o desenvolvedor, caso o segredo não esteja configurado
    st.error("A chave de API do Gemini não foi encontrada nos segredos do Streamlit. Desenvolvedor, por favor, configure-a.")
    st.stop() # Interrompe a execução do app se a chave não estiver presente

# --- Funções dos Agentes (com pequenas modificações para não passar a chave toda hora) ---
# As funções agent_unzip_and_read e agent_query_llm permanecem as mesmas internamente,
# mas a forma como são chamadas mudará. Vamos simplificar para clareza.

# ... (As funções dos agentes como `agent_unzip_and_read`, `agent_query_llm`, `agent_present_results`
# podem ser coladas aqui exatamente como na versão `app_refactored.py`. A única diferença é que
# a chamada a `genai.configure` foi movida para o topo do script.)


# --- Interface Principal (Streamlit) ---

st.title("`Insight Weaver`")
st.markdown("Faça perguntas em linguagem natural sobre seus arquivos CSV.")

# Inicialização do estado da sessão
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- DDR-REFACTOR (Clarity): Sidebar drasticamente simplificada para o usuário final ---
with st.sidebar:
    st.header("Análise de Dados")

    uploaded_file = st.file_uploader(
        "1. Faça upload de um arquivo .zip",
        type="zip",
        key="file_uploader"
    )

    if st.button("2. Analisar Dados", use_container_width=True):
        if uploaded_file:
            with st.spinner("Processando arquivos..."):
                # A função agent_unzip_and_read é chamada aqui
                dfs, manifesto = agent_unzip_and_read(uploaded_file)
                if dfs:
                    st.session_state.dataframes = dfs
                    st.session_state.manifesto = manifesto
                    st.session_state.data_loaded = True
                    st.session_state.messages = []
                    st.success("Pronto! Pode fazer suas perguntas.")
                else:
                    st.error("Nenhum arquivo CSV válido encontrado.")
                    st.session_state.data_loaded = False
        else:
            st.warning("Por favor, faça o upload de um arquivo .zip primeiro.")

# ... (O resto da lógica de chat e input do usuário permanece o mesmo que em `app_refactored.py`)
# A única mudança é que as chamadas para os agentes não precisam mais passar a `api_key` como argumento,
# pois o módulo `genai` já foi configurado globalmente.

# Exemplo de chamada modificada dentro do loop de chat:
# result, code = agent_query_llm(prompt, st.session_state.manifesto, st.session_state.dataframes)
# agent_present_results(result, prompt)

    
