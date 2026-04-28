FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc build-essential libmariadb-dev libmariadb-dev-compat pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root
COPY . /app
EXPOSE 8062
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8062"]
