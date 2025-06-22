# app_fixed.py
# Insight Weaver: Versão pública completa e corrigida.

import streamlit as st
import polars as pl
import google.generativeai as genai
import zipfile
import os
import shutil
from pathlib import Path
import plotly.express as px

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
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, AttributeError):
    st.error("A chave de API do Gemini não foi encontrada. Desenvolvedor, configure o segredo 'GEMINI_API_KEY' nas configurações do Streamlit Cloud.")
    st.stop()

# --- DDR-FIX: 3. Definições Completas das Funções dos Agentes ---
# As definições das funções foram restauradas aqui, antes de serem chamadas.

def agent_unzip_and_read(uploaded_file):
    """Agente Descompactador e de Leitura."""
    temp_dir = Path("./temp_data")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
    if not csv_files:
        return None, None

    dataframes = {}
    data_manifesto = "MANIFESTO DE DADOS DISPONÍVEIS:\n\n"
    for file_name in csv_files:
        df_name = file_name.replace('.csv', '')
        try:
            df = pl.scan_csv(temp_dir / file_name).collect()
            dataframes[df_name] = df
            data_manifesto += f"- Tabela '{df_name}':\n  - Colunas: {df.columns}\n\n"
        except Exception as e:
            st.error(f"Erro ao ler o arquivo {file_name}: {e}")
            return None, None
    return dataframes, data_manifesto

def agent_query_llm(question, manifesto, dfs):
    """Agente de Consulta (O Cérebro)."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Você é um analista de dados sênior especialista na biblioteca Polars em Python.
    Sua tarefa é traduzir a pergunta do usuário em um único bloco de código Python executável.
    **Restrições Estritas:**
    1.  Use SOMENTE a biblioteca Polars.
    2.  Os DataFrames estão em um dicionário chamado `dfs`. Acesse-os como `dfs['nome']`.
    3.  Seu código DEVE atribuir o resultado final a uma variável chamada `result`.
    4.  Responda APENAS com o bloco de código Python. Sem explicações, apenas código.
    **Contexto dos Dados:**
    {manifesto}
    **Pergunta do Usuário:**
    "{question}"
    """
    try:
        response = model.generate_content(prompt)
        code_block = response.text.strip().replace('```python', '').replace('```', '').strip()
        local_scope = {'dfs': dfs, 'pl': pl}
        exec(code_block, {'pl': pl}, local_scope)
        return local_scope.get('result', "Nenhum resultado encontrado."), code_block
    except Exception as e:
        return f"Erro ao executar o código gerado: {e}", None

def agent_present_results(result, question):
    """Agente de Apresentação."""
    if isinstance(result, pl.DataFrame):
        st.dataframe(result.to_pandas(), use_container_width=True)
        try:
            if result.height > 1 and result.width >= 2:
                categorical_col = result.columns[0]
                numerical_col = next((col for col in reversed(result.columns) if result[col].dtype in [pl.Float64, pl.Int64]), None)
                if numerical_col:
                    st.subheader("Visualização Sugerida")
                    fig = px.bar(result.to_pandas(), x=categorical_col, y=numerical_col, title=f"Análise: {question}", text_auto=True)
                    fig.update_layout(title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass # Falha silenciosa se a visualização não for possível
    elif isinstance(result, (int, float, str)):
        st.metric(label="Resultado", value=str(result))
    else:
        st.write(result)

    model = genai.GenerativeModel('gemini-1.5-flash')
    explanation_prompt = f"""
    Baseado na pergunta: "{question}" e no resultado: {result.to_pandas().to_string() if isinstance(result, pl.DataFrame) else str(result)}
    Forneça uma explicação concisa e clara sobre o que esses dados significam, em português.
    """
    explanation_response = model.generate_content(explanation_prompt)
    st.markdown(f"**Análise:**\n{explanation_response.text}")

# --- 4. Lógica Principal da Interface ---

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
            with st.spinner("Processando arquivos..."):
                # Esta chamada agora funciona porque a função foi definida acima.
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message:
            agent_present_results(message["result"], message["original_question"])

if prompt := st.chat_input("Qual a sua pergunta sobre os dados?"):
    if not st.session_state.data_loaded:
        st.warning("Por favor, carregue e analise um arquivo na barra lateral primeiro.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                result, code = agent_query_llm(prompt, st.session_state.manifesto, st.session_state.dataframes)
                if isinstance(result, str) and result.startswith("Erro"):
                    response_content = f"Desculpe, encontrei um erro: {result}"
                    st.error(response_content)
                else:
                    response_content = f"Para responder à sua pergunta, executei a seguinte consulta e obtive estes resultados:"
                    st.markdown(response_content)
                
                st.session_state.messages.append({
                    "role": "assistant", "content": response_content,
                    "result": result, "original_question": prompt
                })
                # Força a re-execução para exibir a nova mensagem imediatamente
                st.rerun()
