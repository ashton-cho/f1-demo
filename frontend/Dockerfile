FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY . .
EXPOSE 8050

CMD ["poetry", "run", "python", "-m", "src.app"]