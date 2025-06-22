# tests/test_app.py
import pytest
from unittest.mock import patch, MagicMock
import polars as pl
from polars.testing import assert_frame_equal
import zipfile
from pathlib import Path

# Importe as funções do seu app.py
# Para isso, seu app.py não deve executar st.set_page_config etc. no nível superior.
# Uma boa prática é colocar a lógica do app dentro de uma função main().
# Por simplicidade aqui, vamos assumir que as funções podem ser importadas.
from app import agent_unzip_and_read, agent_query_llm

@pytest.fixture
def create_test_zip():
    """Cria um arquivo zip de teste para os testes."""
    temp_dir = Path("./test_temp")
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / "test_data.zip"
    csv1_path = temp_dir / "vendas.csv"
    csv2_path = temp_dir / "clientes.csv"

    # Cria CSVs de teste
    csv1_path.write_text("produto,valor\ncarro,50000\naviao,1000000")
    csv2_path.write_text("id_cliente,nome\n1,Joao\n2,Maria")

    # Cria o arquivo ZIP
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(csv1_path, arcname="vendas.csv")
        zf.write(csv2_path, arcname="clientes.csv")

    yield str(zip_path)

    # Limpeza
    csv1_path.unlink()
    csv2_path.unlink()
    zip_path.unlink()
    temp_dir.rmdir()


def test_agent_unzip_and_read(create_test_zip):
    """Testa se o agente descompacta e lê os CSVs corretamente."""
    zip_path = create_test_zip
    with open(zip_path, 'rb') as f:
        dfs, manifesto = agent_unzip_and_read(f)

    assert 'vendas' in dfs
    assert 'clientes' in dfs
    assert "MANIFESTO DE DADOS" in manifesto
    assert "colunas e Tipos: {'produto': String, 'valor': Int64}" in manifesto

    expected_vendas = pl.DataFrame({'produto': ['carro', 'aviao'], 'valor': [50000, 1000000]})
    assert_frame_equal(dfs['vendas'], expected_vendas)

@patch('app.genai.GenerativeModel')
def test_agent_query_llm(MockGenerativeModel):
    """Testa o agente de consulta mockando a resposta do LLM."""
    # Configura o mock
    mock_instance = MockGenerativeModel.return_value
    mock_response = MagicMock()
    mock_response.text = "```python\nresult = dfs['vendas'].filter(pl.col('valor') > 60000)\n```"
    mock_instance.generate_content.return_value = mock_response

    # Dados de entrada
    test_dfs = {'vendas': pl.DataFrame({'produto': ['carro', 'aviao'], 'valor': [50000, 1000000]})}
    question = "Quais vendas foram maiores que 60000?"
    
    result, code = agent_query_llm("fake_api_key", question, "fake_manifesto", test_dfs)

    # Verifica se o código foi extraído corretamente
    assert "result = dfs['vendas'].filter(pl.col('valor') > 60000)" in code
    
    # Verifica se o resultado da execução está correto
    expected_result = pl.DataFrame({'produto': ['aviao'], 'valor': [1000000]})
    assert_frame_equal(result, expected_result)
