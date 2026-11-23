FROM python:3.11-slim
WORKDIR /srv/perturbai
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY app ./app
RUN pip install --no-cache-dir -e . --no-deps
ENV PERTURBAI_DATA=/data
EXPOSE 8501 8000
# mount your data root:  docker run -v $HOME/PerturbAI_data:/data -p 8501:8501 perturbai
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
