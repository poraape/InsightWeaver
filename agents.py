# agents.py
# Este módulo contém a lógica de negócio (os agentes) para a aplicação Insight Weaver.

import streamlit as st
import polars as pl
import google.generativeai as genai
import zipfile
import os
import shutil
from pathlib import Path
import plotly.express as px

# DDR-EXPANSION (Loosely Coupled): Lógica de negócio isolada em seu próprio módulo.

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
            pass
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
