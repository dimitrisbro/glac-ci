FROM python:3.12-slim

WORKDIR /code

COPY poetry.lock pyproject.toml ./

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root \
    && rm -rf $(poetry config cache-dir)/{cache,artifacts}

RUN mkdir output

COPY ./app /code/app

CMD ["python3", "app/main.py"]
