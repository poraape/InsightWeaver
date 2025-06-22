# app_refactored.py
# Insight Weaver: Seu analista de dados conversacional.

import streamlit as st
import polars as pl
import google.generativeai as genai
import zipfile
import os
import shutil
from pathlib import Path
import plotly.express as px

# --- Configuração da Página e Estilo ---
st.set_page_config(
    page_title="Insight Weaver",
    page_icon=" weaver.png",
    layout="centered"
)

st.markdown("""
<style>
    .stApp { background-color: #f0f2f5; }
    .st-emotion-cache-1y4p8pa { padding-top: 2rem; }
    .st-emotion-cache-16txtl3 { padding: 2rem 1rem; }
    h1 { color: #1d1d1f; font-weight: 600; }
    .st-chat-input { background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- Funções dos Agentes (sem alterações) ---

def agent_unzip_and_read(uploaded_file):
    temp_dir = Path("./temp_data")
    # DDR-REFACTOR (Robustness): Limpa o diretório temporário antes de usar.
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
            data_manifesto += f"- Tabela '{df_name}':\n"
            data_manifesto += f"  - Colunas e Tipos: {df.schema}\n"
            data_manifesto += f"  - Amostra de dados:\n{df.head(3).to_pandas().to_string()}\n\n"
        except Exception as e:
            st.error(f"Erro ao ler o arquivo {file_name}: {e}")
            return None, None
    return dataframes, data_manifesto

def agent_query_llm(api_key, question, manifesto, dfs):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Erro de API: {e}", None

    prompt = f"""
    Você é um analista de dados sênior especialista na biblioteca Polars em Python.
    Sua tarefa é traduzir a pergunta do usuário em um único bloco de código Python executável que usa a biblioteca Polars.
    **Restrições Estritas:**
    1.  Use SOMENTE a biblioteca Polars para manipulação de dados.
    2.  Os DataFrames estão pré-carregados e disponíveis em um dicionário Python chamado `dfs`.
    3.  Acesse os dataframes usando a sintaxe `dfs['nome_do_dataframe']`.
    4.  Seu código DEVE atribuir o resultado final a uma variável chamada `result`.
    5.  NÃO inclua `import polars as pl`. A biblioteca já está importada.
    6.  NÃO inclua a definição do dicionário `dfs`. Ele já existe no escopo.
    7.  Responda APENAS com o bloco de código Python. Sem explicações, sem texto adicional, apenas código.
    **Contexto dos Dados:**
    {manifesto}
    **Pergunta do Usuário:**
    "{question}"
    **Exemplo de Resposta Esperada:**
    ```python
    result = dfs['vendas'].filter(pl.col('valor') > 100).group_by('produto').agg(pl.sum('valor')).sort('valor', descending=True)
    ```
    Agora, gere o código Polars para a pergunta do usuário.
    """
    try:
        response = model.generate_content(prompt)
        code_block = response.text.strip().replace('```python', '').replace('```', '').strip()
        local_scope = {'dfs': dfs, 'pl': pl}
        exec(code_block, {'pl': pl}, local_scope)
        return local_scope.get('result', "Nenhum resultado encontrado."), code_block
    except Exception as e:
        return f"Erro ao executar o código gerado: {e}", None

def agent_present_results(api_key, result, question):
    if isinstance(result, pl.DataFrame):
        st.dataframe(result.to_pandas(), use_container_width=True)
        try:
            if result.height > 1 and result.width >= 2:
                categorical_col = result.columns[0]
                numerical_col = next((col for col in reversed(result.columns) if result[col].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]), None)
                if numerical_col:
                    st.subheader("Visualização Sugerida")
                    fig = px.bar(result.to_pandas(), x=categorical_col, y=numerical_col, title=f"Análise: {question}", text_auto=True)
                    fig.update_layout(title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Não foi possível gerar uma visualização automática. {e}")
    elif isinstance(result, (int, float, str)):
        st.metric(label="Resultado", value=str(result))
    else:
        st.write(result)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        explanation_prompt = f"""
        Baseado na pergunta: "{question}" e no resultado: {result.to_pandas().to_string() if isinstance(result, pl.DataFrame) else str(result)}
        Forneça uma explicação concisa e clara sobre o que esses dados significam, em português.
        """
        explanation_response = model.generate_content(explanation_prompt)
        st.markdown(f"**Análise:**\n{explanation_response.text}")
    except Exception as e:
        st.warning(f"Não foi possível gerar a explicação: {e}")

# --- Interface Principal (Streamlit) ---

st.title("`Insight Weaver`")
st.markdown("Faça perguntas em linguagem natural sobre seus arquivos CSV.")

# Inicialização do estado da sessão
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'uploaded_file_obj' not in st.session_state:
    st.session_state.uploaded_file_obj = None

# --- DDR-REFACTOR (Clarity & Robustness): Lógica da sidebar completamente refeita ---
with st.sidebar:
    st.header("Configuração da Análise")
    
    # 1. Entrada da Chave de API
    api_key_input = st.text_input("1. Insira sua Chave de API do Google Gemini", type="password", key="api_key_input")
    if api_key_input:
        st.session_state.api_key = api_key_input

    # 2. Upload do Arquivo
    uploaded_file = st.file_uploader(
        "2. Faça upload de um arquivo .zip com seus CSVs",
        type="zip",
        key="file_uploader"
    )
    if uploaded_file:
        st.session_state.uploaded_file_obj = uploaded_file

    # 3. Botão de Processamento Explícito
    if st.button("Analisar Dados", use_container_width=True):
        # Validação explícita das condições necessárias
        if not st.session_state.api_key:
            st.error("Erro: Chave de API não fornecida.")
        elif not st.session_state.uploaded_file_obj:
            st.error("Erro: Nenhum arquivo .zip foi carregado.")
        else:
            with st.spinner("Processando arquivos..."):
                dfs, manifesto = agent_unzip_and_read(st.session_state.uploaded_file_obj)
                if dfs:
                    st.session_state.dataframes = dfs
                    st.session_state.manifesto = manifesto
                    st.session_state.data_loaded = True
                    st.session_state.messages = [] # Limpa chat anterior
                    st.success("Pronto! Os dados foram analisados. Você já pode fazer perguntas.")
                else:
                    st.error("Nenhum arquivo CSV válido encontrado no .zip.")
                    st.session_state.data_loaded = False

# Exibição do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "code" in message:
            st.code(message["code"], language="python")
        if "result" in message:
            agent_present_results(st.session_state.api_key, message["result"], message["original_question"])

# Input do usuário
if prompt := st.chat_input("Qual a sua pergunta sobre os dados?"):
    if not st.session_state.data_loaded:
        st.warning("Por favor, configure a análise e clique em 'Analisar Dados' na barra lateral primeiro.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                result, code = agent_query_llm(st.session_state.api_key, prompt, st.session_state.manifesto, st.session_state.dataframes)
                if isinstance(result, str) and result.startswith("Erro"):
                    st.error(result)
                    msg_content = f"Desculpe, encontrei um erro: {result}"
                    st.session_state.messages.append({"role": "assistant", "content": msg_content})
                else:
                    response_intro = "Para sua pergunta, executei a seguinte consulta:"
                    st.markdown(response_intro)
                    st.code(code, language="python")
                    agent_present_results(st.session_state.api_key, result, prompt)
                    st.session_state.messages.append({
                        "role": "assistant", "content": response_intro, "code": code,
                        "result": result, "original_question": prompt
                    })
