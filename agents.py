# agents_self_testing.py
# Versão com um Agente Revisor de Qualidade para auto-teste e auto-correção proativa.

import streamlit as st
import polars as pl
import google.generativeai as genai
import zipfile
import os
import shutil
from pathlib import Path
import plotly.express as px
import json

# --- Constantes e Funções de Leitura/Sanitização (Sem alterações) ---
CATEGORICAL_THRESHOLD = 0.5
# ... (cole aqui as funções agent_unzip_and_read, _try_convert_..., agent_sanitize_and_enrich)
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

# --- DDR-EXPANSION: NOVO AGENTE REVISOR DE QUALIDADE ---
def agent_code_reviewer(code_to_review: str, question: str, rules: str) -> dict:
    """
    Este agente atua como um revisor de código sênior. Ele não gera código, apenas o valida.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Você é um revisor de código (code reviewer) sênior, especialista em Polars e extremamente rigoroso.
    Sua tarefa é analisar o bloco de código Python fornecido e verificar se ele atende a todas as regras e se responde corretamente à pergunta do usuário.

    **REGRAS DE VALIDAÇÃO:**
    {rules}

    **PERGUNTA ORIGINAL DO USUÁRIO:**
    "{question}"

    **CÓDIGO PARA REVISÃO:**
    ```python
    {code_to_review}
    ```

    **SUA RESPOSTA:**
    Responda APENAS com um objeto JSON com a seguinte estrutura:
    {{
      "status": "APROVADO" ou "REJEITADO",
      "reason": "Uma explicação concisa da sua decisão. Se rejeitado, explique o erro exato."
    }}
    """
    try:
        response = model.generate_content(prompt)
        # Limpa a resposta para garantir que seja um JSON válido
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response)
    except Exception as e:
        st.warning(f"O Agente Revisor falhou: {e}")
        return {"status": "ERRO_REVISOR", "reason": str(e)}

# --- AGENTE DE CONSULTA (ORQUESTRADOR) - VERSÃO FINAL ---
def agent_query_llm(question, manifesto, dfs, max_retries=1):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    rules = """
    1.  O código deve usar APENAS sintaxe Polars. Nenhuma sintaxe de Pandas é permitida.
    2.  A função de agrupamento deve ser `group_by`, não `groupby`.
    3.  O código deve ser logicamente correto para responder à pergunta do usuário.
    4.  O resultado final deve ser atribuído a uma variável chamada `result`.
    """
    
    prompt_gerador = f"""
    Você é um programador Polars. Gere um bloco de código Polars para responder à pergunta abaixo, seguindo as regras.
    **REGRAS:**
    {rules}
    **MANIFESTO DE DADOS:**
    {manifesto}
    **PERGUNTA:** "{question}"
    **Responda apenas com o bloco de código Python.**
    """
    
    last_error = "Nenhum erro inicial."
    code_block = ""

    for attempt in range(max_retries + 1):
        # --- ETAPA DE GERAÇÃO / CORREÇÃO ---
        if attempt > 0:
            st.warning(f"Tentativa de correção {attempt}...")
            prompt_correcao = f"""
            O código anterior foi rejeitado.
            **Código com Erro:**
            ```python
            {code_block}
            ```
            **Motivo da Rejeição / Erro:**
            {last_error}
            **Sua Tarefa:**
            Reescreva o bloco de código Polars para corrigir o erro, seguindo todas as regras.
            Responda APENAS com o bloco de código corrigido.
            """
            response = model.generate_content(prompt_correcao)
        else:
            response = model.generate_content(prompt_gerador)

        code_block = response.text.strip().replace("```python", "").replace("```", "").strip()

        # --- ETAPA DE AUTO-TESTE (REVISÃO DE CÓDIGO) ---
        st.info("Revisor de Qualidade analisando o código gerado...")
        review = agent_code_reviewer(code_block, question, rules)

        if review.get("status") == "APROVADO":
            st.success("Código APROVADO pelo revisor. Tentando executar...")
            try:
                local_scope = {'dfs': dfs, 'pl': pl}
                exec(code_block, {'pl': pl}, local_scope)
                return local_scope.get('result', "Código executado, mas sem resultado."), code_block
            except Exception as e:
                last_error = f"Erro de Execução: O código foi aprovado, mas falhou ao executar. Erro: {e}"
                # Continua para a próxima tentativa de correção
        else: # REJEITADO ou ERRO_REVISOR
            last_error = f"Rejeitado pelo Revisor: {review.get('reason', 'Motivo não especificado.')}"
            # Continua para a próxima tentativa de correção

    # Se o loop terminar sem sucesso
    return f"Falha ao gerar código funcional após {max_retries + 1} tentativas. Último motivo: {last_error}", code_block

# --- AGENTE 4: APRESENTAÇÃO DE RESULTADOS (Sem alterações) ---
def agent_present_results(result, question):
    # ... (cole aqui a função agent_present_results sem alterações)
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
