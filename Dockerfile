# Dockerfile para a aplicação Insight Weaver

# Use uma imagem base oficial do Python.
FROM python:3.11-slim

# Defina o diretório de trabalho no container.
WORKDIR /app

# Copie o arquivo de dependências primeiro para aproveitar o cache do Docker.
COPY requirements.txt .

# Instale as dependências.
RUN pip install --no-cache-dir -r requirements.txt

# Copie o resto do código da aplicação para o diretório de trabalho.
COPY . .

# Exponha a porta que o Streamlit usa.
EXPOSE 8501

# Comando para executar a aplicação quando o container iniciar.
# O healthcheck garante que o serviço de deploy saiba que o app está rodando.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
