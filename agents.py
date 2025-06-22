# agents_final_bulletproof.py
# Versão final com tratamento de exceção universal para máxima robustez.

import streamlit as st
import polars as pl
import google.generativeai as genai
import zipfile
import os
import shutil
from pathlib import Path
import plotly.express as px

# --- Constantes de configuração ---
CATEGORICAL_THRESHOLD = 0.5

# --- AGENTE 1: LEITURA BRUTA (Sem alterações) ---
def agent_unzip_and_read(uploaded_file):
    temp_dir = Path("./temp_data")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
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

# --- Funções de Conversão Atômicas (com tratamento de exceção aprimorado) ---

def _try_convert_to_numeric(series: pl.Series) -> pl.Series | None:
    """Tenta converter uma série de String para Float64."""
    # DDR-FIX (Robustness): Captura qualquer exceção durante a conversão.
    try:
        # Tenta a conversão estrita. Se falhar, a exceção é capturada.
        return series.str.replace_all(",", ".", literal=True).cast(pl.Float64, strict=True)
    except Exception:
        # Se qualquer erro ocorrer, a conversão falhou. Retorna None.
        return None

def _try_convert_to_date(series: pl.Series) -> pl.Series | None:
    """Tenta converter uma série de String para Date usando múltiplos formatos."""
    # DDR-FIX (Robustness): Captura qualquer exceção durante a conversão.
    try:
        # Tenta a conversão estrita. Se falhar, a exceção é capturada.
        return series.str.to_date(formats=["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"], strict=True)
    except Exception:
        # Se qualquer erro ocorrer, a conversão falhou. Retorna None.
        return None

def _try_convert_to_categorical(series: pl.Series) -> pl.Series | None:
    """Converte para Categórico se a cardinalidade for baixa."""
    # Esta operação é segura e não precisa de try/except.
    if series.n_unique() / len(series) < CATEGORICAL_THRESHOLD:
        return series.cast(pl.Categorical)
    return None

# --- AGENTE 2: SANITIZAÇÃO E ENRIQUECIMENTO (Sem alterações na lógica de orquestração) ---
def agent_sanitize_and_enrich(dataframes: dict):
    sanitized_dfs = {}
    data_manifesto = "MANIFESTO DE DADOS DISPONÍVEIS (Após limpeza e otimização):\n\n"

    for name, df in dataframes.items():
        sanitized_df = df.clone()
        
        for col_name in df.columns:
            original_series = sanitized_df[col_name]

            if original_series.dtype == pl.String:
                clean_series = original_series.str.strip_chars()
                
                numeric_series = _try_convert_to_numeric(clean_series)
                if numeric_series is not None:
                    final_series = numeric_series
                else:
                    date_series = _try_convert_to_date(clean_series)
                    if date_series is not None:
                        final_series = date_series
                    else:
                        categorical_series = _try_convert_to_categorical(clean_series)
                        if categorical_series is not None:
                            final_series = categorical_series
                        else:
                            final_series = clean_series
                
                sanitized_df = sanitized_df.with_columns(final_series.alias(col_name))

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
