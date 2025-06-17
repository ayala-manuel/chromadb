FROM python:3.10-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

RUN mkdir -p /app/vectorstore

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--reload"]
