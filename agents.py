# agents_final_debuggable.py
# Versão com logging de erro no agente de sugestões para facilitar a depuração.

import streamlit as st
import polars as pl
import google.generativeai as genai
import zipfile
import os
import shutil
from pathlib import Path
import plotly.express as px
import ast # Usaremos ast.literal_eval para uma avaliação mais segura

# --- Constantes de configuração ---
CATEGORICAL_THRESHOLD = 0.5

# --- DDR-FIX (Clarity & Robustness): AGENTE 1.5: SUGESTÃO DE PERGUNTAS (com logging) ---
def agent_suggest_questions(manifesto: str):
    """Usa o manifesto de dados para sugerir perguntas de negócio inteligentes."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Você é um Analista de Negócios Sênior. Sua tarefa é analisar o seguinte manifesto de dados e propor 3 perguntas de negócio perspicazes que podem ser respondidas com os dados disponíveis.

    **REGRAS CRÍTICAS:**
    1.  As perguntas devem ser claras, concisas e orientadas a insights.
    2.  Sua resposta DEVE ser uma lista Python de strings, formatada como um bloco de código. NADA MAIS.
    3.  Exemplo de Resposta:
    ```python
    ["Qual foi o total de vendas por mês?", "Quais são os 5 principais clientes por valor de compra?", "Qual a margem de lucro por categoria de produto?"]
    ```

    **MANIFESTO DE DADOS:**
    {manifesto}

    **Gere a lista de 3 perguntas agora:**
    """
    try:
        response = model.generate_content(prompt)
        
        # Limpa a resposta para extrair apenas o conteúdo da lista
        clean_response = response.text.strip()
        if "```python" in clean_response:
            clean_response = clean_response.split("```python")[1]
        if "```" in clean_response:
            clean_response = clean_response.split("```")[0]
        
        # Usa ast.literal_eval que é mais seguro que eval()
        suggested_list = ast.literal_eval(clean_response.strip())
        
        if isinstance(suggested_list, list):
            return suggested_list
        
        # Se não for uma lista, loga um aviso
        st.warning(f"O agente de sugestões retornou um tipo inesperado: {type(suggested_list)}")
        return []
        
    except Exception as e:
        # Se qualquer erro ocorrer, loga o erro na interface para depuração
        st.warning(f"Não foi possível gerar sugestões de perguntas. Erro: {e}")
        st.info(f"Resposta bruta da IA (para depuração): {response.text}")
        return []

# ... (O resto do código de agents.py permanece o mesmo) ...
def agent_unzip_and_read(uploaded_file):
    temp_dir = Path("./temp_data")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    csv_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)
    if not csv_files: return None
    dataframes = {}
    for file_path in csv_files:
        df_name = file_path.stem
        try:
            df = pl.read_csv(source=file_path, has_header=True, infer_schema_length=0, ignore_errors=True)
            dataframes[df_name] = df
        except Exception as e:
            st.error(f"Erro crítico ao ler o arquivo {file_path.name}: {e}")
            return None
    return dataframes

def _try_convert_to_numeric(series: pl.Series) -> pl.Series | None:
    try:
        return series.str.replace_all(",", ".", literal=True).cast(pl.Float64, strict=True)
    except Exception:
        return None

def _try_convert_to_date(series: pl.Series) -> pl.Series | None:
    try:
        return series.str.to_date(formats=["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"], strict=True)
    except Exception:
        return None

def _try_convert_to_categorical(series: pl.Series) -> pl.Series | None:
    if series.n_unique() / len(series) < CATEGORICAL_THRESHOLD:
        return series.cast(pl.Categorical)
    return None

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
        data_manifesto += f"- Tabela '{name}':\n  - Colunas e Tipos: {sanitized_df.schema}\n"
        data_manifesto += f"  - Amostra de dados:\n{sanitized_df.head(3).to_pandas().to_string()}\n\n"
    return sanitized_dfs, data_manifesto

def agent_query_llm(question, manifesto, dfs, max_retries=1):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Você é um assistente de programação especialista EXCLUSIVAMENTE na biblioteca Polars para Python. Sua única tarefa é traduzir a pergunta do usuário em um bloco de código Polars executável.
    **REGRAS CRÍTICAS:**
    1.  **USE APENAS SINTAXE POLARS.**
    2.  **ERRO COMUM A EVITAR:** A sintaxe de agrupamento em Polars é `group_by`. **NÃO USE `groupby` (de Pandas).**
    3.  Os DataFrames estão em um dicionário `dfs`.
    4.  O resultado final DEVE ser atribuído a uma variável `result`.
    5.  Responda APENAS com o bloco de código Python.
    **EXEMPLO:**
    - Pergunta: "qual o cliente que mais comprou em valor?"
    - Resposta:
    ```
result = dfs['vendas'].group_by('nome_cliente').agg(
    pl.sum('valor_compra').alias('total_comprado')
).sort('total_comprado', descending=True).limit(1)
    ```
    **MANIFESTO DE DADOS:**
    {manifesto}
    **TAREFA ATUAL:**
    - Pergunta: "{question}"
    - Gere o código Polars:
    """
    last_error = None
    code_block = ""
    for attempt in range(max_retries + 1):
        if attempt > 0:
            st.warning(f"Tentativa {attempt}: O código anterior falhou. Tentando corrigir...")
            correction_prompt = f"""
            O código que você gerou anteriormente falhou.
            **Código com Erro:**
            ```python
            {code_block}
            ```
            **Mensagem de Erro:**
            {last_error}
            **Sua Tarefa:**
            Analise o erro e o código. Reescreva o bloco de código Polars para corrigir o erro, seguindo todas as regras originais.
            Responda APENAS com o bloco de código Python corrigido.
            """
            response = model.generate_content(correction_prompt)
        else:
            response = model.generate_content(prompt)
        code_block = response.text.strip()
        if code_block.startswith("```python"): code_block = code_block[9:]
        elif code_block.startswith("```"): code_block = code_block[3:]
        if code_block.endswith("```"): code_block = code_block[:-3]
        code_block = code_block.strip()
        if ".groupby(" in code_block:
            last_error = "Erro Heurístico: Sintaxe proibida de Pandas '.groupby' detectada."
            continue
        try:
            local_scope = {'dfs': dfs, 'pl': pl}
            exec(code_block, {'pl': pl}, local_scope)
            return local_scope.get('result', "Código executado, mas sem resultado."), code_block
        except Exception as e:
            last_error = str(e)
    return f"Falha ao gerar código funcional após {max_retries + 1} tentativas. Último erro: {last_error}", code_block

def agent_present_results(result, question):
    if isinstance(result, pl.DataFrame):
        st.dataframe(result.to_pandas(), use_container_width=True)
        try:
            if result.height > 1 and result.width >= 2:
                categorical_col = result.columns
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
