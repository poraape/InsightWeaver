# agents_fixed_v2.py
# Versão com o Agente de Sanitização corrigido para evitar o TypeError.

import streamlit as st
import polars as pl
import google.generativeai as genai
import zipfile
import os
import shutil
from pathlib import Path
import plotly.express as px

# --- AGENTE 1: LEITURA BRUTA (Sem alterações) ---
def agent_unzip_and_read(uploaded_file):
    temp_dir = Path("./temp_data")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
    if not csv_files: return None
    dataframes = {}
    for file_name in csv_files:
        df_name = file_name.replace('.csv', '')
        try:
            df = pl.read_csv(source=temp_dir / file_name, has_header=True, infer_schema_length=0, ignore_errors=True)
            dataframes[df_name] = df
        except Exception as e:
            st.error(f"Erro crítico ao ler o arquivo {file_name}: {e}")
            return None
    return dataframes

# --- DDR-FIX (Robustness): AGENTE 2: SANITIZAÇÃO E ENRIQUECIMENTO (Lógica Corrigida) ---
def agent_sanitize_and_enrich(dataframes: dict):
    sanitized_dfs = {}
    data_manifesto = "MANIFESTO DE DADOS DISPONÍVEIS (Após limpeza e otimização):\n\n"

    for name, df in dataframes.items():
        sanitized_df = df.clone()
        
        # Itera sobre os nomes das colunas originais para evitar problemas com modificações no loop
        for col_name in df.columns:
            # Apenas tenta converter colunas que são do tipo String
            if sanitized_df[col_name].dtype == pl.String:
                
                # 1. Limpeza: Sempre segura para colunas de string
                sanitized_df = sanitized_df.with_columns(
                    pl.col(col_name).str.strip_chars().alias(col_name)
                )
                
                # 2. Tenta a conversão para numérico PRIMEIRO
                numeric_col = pl.col(col_name).str.replace_all(",", ".", literal=True).cast(pl.Float64, strict=False)
                # Verifica se a conversão foi bem-sucedida
                if numeric_col.is_not_null().sum() > 0:
                    sanitized_df = sanitized_df.with_columns(numeric_col)
                
                # 3. Se a coluna AINDA for String (falhou na conversão numérica), tenta converter para data
                elif sanitized_df[col_name].dtype == pl.String:
                    date_col = pl.col(col_name).str.to_date(formats=["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"], strict=False)
                    # Verifica se a conversão foi bem-sucedida
                    if date_col.is_not_null().sum() > 0:
                        sanitized_df = sanitized_df.with_columns(date_col)

        sanitized_dfs[name] = sanitized_df
        data_manifesto += f"- Tabela '{name}':\n"
        data_manifesto += f"  - Colunas e Tipos: {sanitized_df.schema}\n"
        data_manifesto += f"  - Amostra de dados:\n{sanitized_df.head(3).to_pandas().to_string()}\n\n"

    return sanitized_dfs, data_manifesto

# --- AGENTE 3: CONSULTA COM IA (Sem alterações) ---
def agent_query_llm(question, manifesto, dfs):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Você é um analista de dados sênior especialista na biblioteca Polars em Python.
    Sua tarefa é traduzir a pergunta do usuário em um único bloco de código Python executável.
    **Contexto Importante:** Os dados já passaram por um processo de limpeza e conversão de tipos. Você pode confiar nos tipos de dados descritos no manifesto abaixo.
    **Restrições Estritas:**
    1.  Use SOMENTE a biblioteca Polars.
    2.  Os DataFrames estão em um dicionário chamado `dfs`. Acesse-os como `dfs['nome']`.
    3.  Seu código DEVE atribuir o resultado final a uma variável chamada `result`.
    4.  Responda APENAS com o bloco de código Python.
    **Contexto dos Dados (Manifesto):**
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

# --- AGENTE 4: APRESENTAÇÃO DE RESULTADOS (Sem alterações) ---
def agent_present_results(result, question):
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
        except Exception: pass
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
